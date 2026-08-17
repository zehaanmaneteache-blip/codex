//! Session, resume, fork, and subagent selection lifecycle for the TUI app.
//!
//! This module owns the high-level transitions between app-server threads: starting fresh sessions,
//! resuming/forking saved sessions, replacing ChatWidget instances, and maintaining the agent picker
//! cache used for multi-agent navigation.

use super::*;
use crate::app_server_session::source_agent_path;
use crate::app_server_session::thread_blocks_direct_input;
use codex_config::types::ResumeCwdMode;
use std::collections::HashMap;
use std::collections::HashSet;

const SUBAGENT_BACKFILL_PAGE_SIZE: u32 = 100;
const MAX_RETIRED_THREAD_IDS: usize = 512;

#[derive(Clone, Copy)]
pub(super) enum ThreadAttachPresentation {
    SessionLineage,
    PromptEdit,
}

/// Reports whether a loaded-thread backfill completed and which descendants already had their
/// liveness metadata refreshed, allowing the picker to skip duplicate `thread/read` requests.
#[derive(Default)]
pub(super) struct LoadedSubagentBackfill {
    pub(super) completed: bool,
    pub(super) refreshed_thread_ids: HashSet<ThreadId>,
    pub(super) discovered_thread_ids: HashSet<ThreadId>,
}

impl App {
    pub(super) fn retire_thread_id(&mut self, thread_id: ThreadId) {
        if self.retired_thread_ids.insert(thread_id) {
            self.retired_thread_order.push_back(thread_id);
        }
        while self.retired_thread_order.len() > MAX_RETIRED_THREAD_IDS {
            let Some(oldest_thread_id) = self.retired_thread_order.pop_front() else {
                break;
            };
            self.retired_thread_ids.remove(&oldest_thread_id);
        }
    }

    pub(super) fn unretire_thread_id(&mut self, thread_id: ThreadId) {
        self.retired_thread_ids.remove(&thread_id);
        self.retired_thread_order
            .retain(|candidate| *candidate != thread_id);
    }

    pub(super) async fn open_agent_picker(&mut self, app_server: &mut AppServerSession) {
        let backfill = self.backfill_loaded_subagent_threads(app_server).await;
        self.reconcile_agent_navigation_after_backfill(&backfill);
        // V2 subagents are identified by canonical paths observed from activity events or loaded
        // thread metadata. A buffered active turn is positive liveness evidence; a completed
        // snapshot is terminal evidence. An empty store does not clear a successful spawn hint.
        let path_backed_thread_ids: Vec<_> = self
            .agent_navigation
            .ordered_path_backed_subagent_threads(self.primary_thread_id)
            .into_iter()
            .map(|(thread_id, _)| thread_id)
            .collect();
        for thread_id in path_backed_thread_ids.iter().copied() {
            if let Some(channel) = self.thread_event_channels.get(&thread_id)
                && channel.attachment() == ThreadEventAttachment::Live
            {
                let (has_active_turn, has_terminal_snapshot) = {
                    let store = channel.store.lock().await;
                    (
                        store.active_turn_id().is_some(),
                        store
                            .turns
                            .last()
                            .is_some_and(|turn| !matches!(turn.status, TurnStatus::InProgress)),
                    )
                };
                if has_active_turn {
                    self.agent_navigation.mark_running(thread_id);
                } else if has_terminal_snapshot {
                    self.agent_navigation.mark_stopped(thread_id);
                }
            } else if !backfill.refreshed_thread_ids.contains(&thread_id) {
                self.refresh_agent_picker_thread_liveness(app_server, thread_id)
                    .await;
            }
        }
        let path_backed_threads = self
            .agent_navigation
            .ordered_path_backed_subagent_threads(self.primary_thread_id);
        if !path_backed_threads.is_empty() {
            let running_threads: Vec<_> = path_backed_threads
                .into_iter()
                .filter_map(|(thread_id, entry)| {
                    if !entry.is_running || entry.is_closed {
                        return None;
                    }
                    Some((thread_id, entry.agent_path.as_deref()?.trim().to_string()))
                })
                .collect();
            let mut entries = Vec::new();
            for (thread_id, agent_path) in running_threads {
                let preview = if let Some(channel) = self.thread_event_channels.get(&thread_id) {
                    let store = channel.store.lock().await;
                    super::agent_status_feed::AgentStatusThreadPreview::from_store(
                        agent_path, &store,
                    )
                } else {
                    super::agent_status_feed::AgentStatusThreadPreview::empty(agent_path)
                };
                entries.push(preview);
            }

            self.chat_widget
                .add_to_history(super::agent_status_feed::AgentStatusHistoryCell::new(
                    entries,
                ));
        }

        let mut thread_ids = self.agent_navigation.tracked_thread_ids();
        for thread_id in self.thread_event_channels.keys().copied() {
            if !thread_ids.contains(&thread_id) {
                thread_ids.push(thread_id);
            }
        }
        for thread_id in thread_ids {
            if path_backed_thread_ids.contains(&thread_id)
                || self.side_threads.contains_key(&thread_id)
                || backfill.refreshed_thread_ids.contains(&thread_id)
            {
                continue;
            }
            if !self
                .refresh_agent_picker_thread_liveness(app_server, thread_id)
                .await
            {
                continue;
            }
        }

        let has_non_primary_agent_thread = self
            .agent_navigation
            .has_non_primary_thread(self.primary_thread_id);
        if !self.config.features.enabled(Feature::Collab) && !has_non_primary_agent_thread {
            self.chat_widget.open_multi_agent_enable_prompt();
            return;
        }

        if self.agent_navigation.is_empty() {
            self.chat_widget
                .add_info_message("No agents available yet.".to_string(), /*hint*/ None);
            return;
        }

        let mut initial_selected_idx = None;
        let items: Vec<SelectionItem> = self
            .agent_navigation
            .ordered_threads()
            .into_iter()
            .enumerate()
            .map(|(idx, (thread_id, entry))| {
                if self.active_thread_id == Some(thread_id) {
                    initial_selected_idx = Some(idx);
                }
                let id = thread_id;
                let is_primary = self.primary_thread_id == Some(thread_id);
                let name = entry
                    .agent_path
                    .as_deref()
                    .map(str::trim)
                    .filter(|agent_path| !is_primary && !agent_path.is_empty())
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| {
                        format_agent_picker_item_name(
                            entry.agent_nickname.as_deref(),
                            entry.agent_role.as_deref(),
                            is_primary,
                        )
                    });
                let uuid = thread_id.to_string();
                SelectionItem {
                    name: name.clone(),
                    name_prefix_spans: agent_picker_status_dot_spans(entry.is_closed),
                    description: Some(uuid.clone()),
                    is_current: self.active_thread_id == Some(thread_id),
                    actions: vec![Box::new(move |tx| {
                        tx.send(AppEvent::SelectAgentThread(id));
                    })],
                    dismiss_on_select: true,
                    search_value: Some(format!("{name} {uuid}")),
                    ..Default::default()
                }
            })
            .collect();

        self.chat_widget.show_selection_view(SelectionViewParams {
            title: Some("Subagents".to_string()),
            subtitle: Some(AgentNavigationState::picker_subtitle()),
            footer_hint: Some(standard_popup_hint_line()),
            items,
            initial_selected_idx,
            ..Default::default()
        });
    }

    pub(super) fn is_terminal_thread_read_error(err: &color_eyre::Report) -> bool {
        err.chain().any(|cause| {
            let message = cause.to_string().to_ascii_lowercase();
            message.contains("thread not found:")
                || message.contains("thread does not exist:")
                || message.contains("no rollout found for thread id")
        })
    }

    pub(super) fn closed_state_for_thread_read_error(
        err: &color_eyre::Report,
        existing_is_closed: Option<bool>,
    ) -> bool {
        Self::is_terminal_thread_read_error(err) || existing_is_closed.unwrap_or(false)
    }

    pub(super) fn can_fallback_from_include_turns_error(err: &color_eyre::Report) -> bool {
        err.chain().any(|cause| {
            let message = cause.to_string();
            message.contains("includeTurns is unavailable before first user message")
                || message.contains("ephemeral threads do not support includeTurns")
        })
    }

    /// Updates cached picker metadata and then mirrors any visible-label change into the footer.
    ///
    /// These two writes stay paired so the picker rows and contextual footer continue to describe
    /// the same displayed thread after nickname or role updates.
    pub(super) fn upsert_agent_picker_thread(
        &mut self,
        thread_id: ThreadId,
        agent_nickname: Option<String>,
        agent_role: Option<String>,
        is_closed: bool,
    ) {
        // An entry carried across a same-primary resume is only a fallback until new local evidence
        // observes it again. Do not let a complete backfill prune a child whose event arrived after
        // the snapshot began merely because its id was already present in the carried set.
        self.carried_navigation_thread_ids.remove(&thread_id);
        // Backfill responses can omit optional agent metadata even though the picker already has
        // a richer entry from an earlier activity event. Treat omitted fields as unchanged so a
        // partial or metadata-only refresh cannot erase the navigation label while restoring the
        // thread tree.
        let existing_entry = self.agent_navigation.get(&thread_id).cloned();
        let agent_nickname = agent_nickname.or_else(|| {
            existing_entry
                .as_ref()
                .and_then(|entry| entry.agent_nickname.clone())
        });
        let agent_role = agent_role.or_else(|| {
            existing_entry
                .as_ref()
                .and_then(|entry| entry.agent_role.clone())
        });
        self.chat_widget.set_collab_agent_metadata(
            thread_id,
            agent_nickname.clone(),
            agent_role.clone(),
        );
        self.agent_navigation
            .upsert(thread_id, agent_nickname, agent_role, is_closed);
        self.sync_active_agent_label();
    }

    /// Applies the app-server's authoritative direct-input policy to the primary thread.
    pub(super) fn set_primary_thread_parent_owned(
        &mut self,
        thread_id: ThreadId,
        parent_owned: bool,
    ) {
        self.agent_navigation
            .set_parent_owned(thread_id, parent_owned);
        self.chat_widget.set_direct_input_blocked(parent_owned);
    }

    /// Marks a cached picker thread closed and recomputes the contextual footer label.
    ///
    /// Closing a thread is not the same as removing it: users can still inspect finished agent
    /// transcripts, and the stable next/previous traversal order should not collapse around them.
    pub(super) fn mark_agent_picker_thread_closed(&mut self, thread_id: ThreadId) {
        self.agent_navigation.mark_closed(thread_id);
        self.sync_active_agent_label();
    }

    pub(super) async fn refresh_agent_picker_thread_liveness(
        &mut self,
        app_server: &mut AppServerSession,
        thread_id: ThreadId,
    ) -> bool {
        let existing_entry = self.agent_navigation.get(&thread_id).cloned();
        let has_replay_channel = self.thread_event_channels.contains_key(&thread_id);
        match app_server
            .thread_read(thread_id, /*include_turns*/ false)
            .await
        {
            Ok(thread) => {
                // A successful read is positive evidence that this id is still attachable. This
                // also releases a child that was retained across a partial backfill and can now be
                // selected successfully.
                self.unretire_thread_id(thread_id);
                // A liveness read proves that the record is readable, not that it was present in
                // the authoritative descendant snapshot. Keep the carry marker until a complete
                // backfill explicitly discovers or prunes the entry; otherwise a picker refresh
                // after a partial list failure would accidentally make the stale row unprunable.
                let was_carried = self.carried_navigation_thread_ids.contains(&thread_id);
                let is_parent_owned = thread_blocks_direct_input(&thread);
                let agent_path = source_agent_path(&thread.source);
                let is_running = matches!(
                    thread.status,
                    codex_app_server_protocol::ThreadStatus::Active { .. }
                );
                // A successful read proves that the persisted thread is resumable even when its
                // runtime status is `NotLoaded`.
                let is_closed = false;
                self.upsert_agent_picker_thread(
                    thread_id,
                    thread.agent_nickname.or_else(|| {
                        existing_entry
                            .as_ref()
                            .and_then(|entry| entry.agent_nickname.clone())
                    }),
                    thread.agent_role.or_else(|| {
                        existing_entry
                            .as_ref()
                            .and_then(|entry| entry.agent_role.clone())
                    }),
                    is_closed,
                );
                self.agent_navigation
                    .set_parent_owned(thread_id, is_parent_owned);
                self.agent_navigation.set_agent_path(thread_id, agent_path);
                if was_carried {
                    self.carried_navigation_thread_ids.insert(thread_id);
                }
                if is_running {
                    self.agent_navigation.mark_running(thread_id);
                } else {
                    self.agent_navigation
                        .set_running(thread_id, /*is_running*/ false);
                }
                true
            }
            Err(err) => {
                // A transient `thread not loaded` response is recoverable: the persisted record
                // may still be visible through the next picker/backfill pass. Keep the entry and
                // let selection retry instead of turning a runtime transition into deletion.
                if Self::is_terminal_thread_read_error(&err) && !has_replay_channel {
                    self.carried_navigation_thread_ids.remove(&thread_id);
                    self.retire_thread_id(thread_id);
                    self.agent_navigation.remove(thread_id);
                    return false;
                }
                let was_carried = self.carried_navigation_thread_ids.contains(&thread_id);
                let is_terminal_error = Self::is_terminal_thread_read_error(&err);
                let is_closed = Self::closed_state_for_thread_read_error(
                    &err,
                    existing_entry.as_ref().map(|entry| entry.is_closed),
                );
                if let Some(entry) = existing_entry {
                    self.upsert_agent_picker_thread(
                        thread_id,
                        entry.agent_nickname,
                        entry.agent_role,
                        is_closed,
                    );
                } else {
                    self.upsert_agent_picker_thread(
                        thread_id, /*agent_nickname*/ None, /*agent_role*/ None,
                        is_closed,
                    );
                }
                self.agent_navigation
                    .set_running(thread_id, /*is_running*/ false);
                // A failed read is not positive evidence that a carried entry belongs to the
                // current authoritative snapshot. Keep the carry marker for transient failures so
                // a later complete backfill can still reconcile the entry. A terminal error with a
                // replay channel is different: the local channel itself is evidence to retain.
                if was_carried && !is_terminal_error {
                    self.carried_navigation_thread_ids.insert(thread_id);
                }
                true
            }
        }
    }

    /// Materializes a live thread into local replay state when the picker knows about it but the
    /// TUI has not cached a local event channel yet.
    ///
    /// Resume-time backfill intentionally avoids creating empty placeholder channels, because those
    /// placeholders make stale `/agent` entries open blank transcripts. When a user later selects a
    /// still-live discovered thread, attach it on demand with a real resumed snapshot. A
    /// `ReplayOnly` channel follows the same path so a later successful resume can restore its live
    /// attachment.
    pub(super) async fn attach_live_thread_for_selection(
        &mut self,
        app_server: &mut AppServerSession,
        thread_id: ThreadId,
    ) -> Result<bool> {
        let has_live_channel = self
            .thread_event_channels
            .get(&thread_id)
            .is_some_and(|channel| channel.attachment() == ThreadEventAttachment::Live);
        if has_live_channel {
            return Ok(true);
        }

        let (session, turns, live_attached) = match app_server
            .resume_thread(self.config.clone(), thread_id, self.resume_model_settings())
            .await
        {
            Ok(started) => {
                self.unretire_thread_id(thread_id);
                self.agent_navigation
                    .set_parent_owned(thread_id, started.blocks_direct_input);
                (started.session, started.turns, true)
            }
            Err(resume_err) => {
                tracing::warn!(
                    thread_id = %thread_id,
                    error = %resume_err,
                    "failed to resume live thread for selection; falling back to thread/read"
                );
                let (thread, turns) = match app_server
                    .thread_read(thread_id, /*include_turns*/ true)
                    .await
                {
                    Ok(thread) => {
                        self.unretire_thread_id(thread_id);
                        let turns = thread.turns.clone();
                        (thread, turns)
                    }
                    Err(err) if Self::can_fallback_from_include_turns_error(&err) => {
                        let thread = app_server
                            .thread_read(thread_id, /*include_turns*/ false)
                            .await?;
                        self.unretire_thread_id(thread_id);
                        (thread, Vec::new())
                    }
                    Err(err) => return Err(err),
                };
                if turns.is_empty() {
                    // A `thread/read` fallback without turns would create a blank local replay
                    // channel with no live listener attached, which blocks later real re-attach.
                    return Err(color_eyre::eyre::eyre!(
                        "Agent thread {thread_id} is not yet available for replay or live attach."
                    ));
                }
                let mut session = self.session_state_for_thread_read(thread_id, &thread).await;
                // `thread/read` can seed replay state, but it does not attach the app-server
                // listener that `thread/resume` establishes, so treat this path as replay-only.
                session.model.clear();
                (session, turns, false)
            }
        };
        let channel = self.ensure_thread_channel(thread_id);
        if live_attached {
            channel.mark_live();
        } else {
            channel.mark_replay_only();
        }
        let mut store = channel.store.lock().await;
        store.set_session(session, turns);
        Ok(live_attached)
    }

    /// Replaces the chat widget and re-seeds the new widget's collab metadata from the navigation
    /// cache.
    ///
    /// Thread switches reconstruct the `ChatWidget`, which loses the `collab_agent_metadata` map.
    /// This helper copies every known nickname/role from `AgentNavigationState` into the
    /// replacement widget so that replayed collab items render agent names immediately.
    pub(super) fn replace_chat_widget(&mut self, mut chat_widget: ChatWidget) {
        // Transfer the last-written terminal title to the replacement widget
        // so it knows what OSC title is currently displayed. Without this, the
        // new widget would redundantly clear and rewrite the same title, causing
        // a visible flicker in some terminals.
        let previous_terminal_title = self.chat_widget.last_terminal_title.take();
        if chat_widget.last_terminal_title.is_none() {
            chat_widget.last_terminal_title = previous_terminal_title;
        }
        chat_widget.remote_connection = self.chat_widget.remote_connection.clone();
        for (thread_id, entry) in self.agent_navigation.ordered_threads() {
            chat_widget.set_collab_agent_metadata(
                thread_id,
                entry.agent_nickname.clone(),
                entry.agent_role.clone(),
            );
        }
        self.chat_widget = chat_widget;
        self.sync_active_agent_label();
    }

    pub(super) async fn select_agent_thread(
        &mut self,
        tui: &mut tui::Tui,
        app_server: &mut AppServerSession,
        thread_id: ThreadId,
    ) -> Result<()> {
        // Re-selecting a live thread is a no-op, but re-selecting a replay-only thread is the
        // retry path that can upgrade it after its original writer has exited.
        if self.active_thread_id == Some(thread_id)
            && !self
                .thread_event_channels
                .get(&thread_id)
                .is_some_and(|channel| channel.attachment() == ThreadEventAttachment::ReplayOnly)
        {
            return Ok(());
        }

        // A tracked side thread stays loaded until it is explicitly discarded and already has a
        // replay channel, so another liveness read cannot add anything before selection.
        if !(self.side_threads.contains_key(&thread_id)
            && self.thread_event_channels.contains_key(&thread_id)
            || self
                .refresh_agent_picker_thread_liveness(app_server, thread_id)
                .await)
        {
            self.chat_widget
                .add_error_message(format!("Agent thread {thread_id} is no longer available."));
            return Ok(());
        }
        let mut is_replay_only = self
            .agent_navigation
            .get(&thread_id)
            .is_some_and(|entry| entry.is_closed);
        let mut attached_replay_only = false;
        if self.should_attach_live_thread_for_selection(thread_id) {
            match self
                .attach_live_thread_for_selection(app_server, thread_id)
                .await
            {
                Ok(live_attached) => {
                    attached_replay_only = !live_attached;
                    if attached_replay_only {
                        is_replay_only = true;
                    }
                }
                Err(err) => {
                    self.chat_widget.add_error_message(format!(
                        "Failed to attach to agent thread {thread_id}: {err}"
                    ));
                    return Ok(());
                }
            }
        } else if !self.thread_event_channels.contains_key(&thread_id) && is_replay_only {
            self.chat_widget
                .add_error_message(format!("Agent thread {thread_id} is no longer available."));
            return Ok(());
        }
        let previous_thread_id = self.active_thread_id;
        self.store_active_thread_receiver().await;
        self.active_thread_id = None;
        let Some((receiver, mut snapshot)) = self.activate_thread_for_replay(thread_id).await
        else {
            self.chat_widget
                .add_error_message(format!("Agent thread {thread_id} is already active."));
            if let Some(previous_thread_id) = previous_thread_id {
                self.activate_thread_channel(previous_thread_id).await;
            }
            return Ok(());
        };

        self.refresh_snapshot_session_if_needed(
            app_server,
            thread_id,
            is_replay_only,
            &mut snapshot,
        )
        .await;
        let blocks_direct_input = self.agent_navigation.is_parent_owned(thread_id);

        self.active_thread_id = Some(thread_id);
        self.active_thread_rx = Some(receiver);

        let init = self.chatwidget_init_for_forked_or_resumed_thread(
            tui,
            self.config.clone(),
            /*initial_user_message*/ None,
        );
        self.replace_chat_widget(ChatWidget::new_with_app_event(init));
        self.chat_widget
            .set_direct_input_blocked(blocks_direct_input);

        self.reset_for_thread_switch(tui)?;
        self.replay_thread_snapshot(snapshot, !is_replay_only);
        if is_replay_only {
            let message = if attached_replay_only {
                format!(
                    "Agent thread {thread_id} could not be resumed live. Replaying saved transcript."
                )
            } else {
                format!("Agent thread {thread_id} is closed. Replaying saved transcript.")
            };
            self.chat_widget.add_info_message(message, /*hint*/ None);
        }
        self.drain_active_thread_events(tui).await?;
        self.refresh_pending_thread_approvals().await;

        Ok(())
    }

    pub(super) fn should_attach_live_thread_for_selection(&self, thread_id: ThreadId) -> bool {
        // A replay-only channel is explicitly retryable: the earlier resume may have failed only
        // because another writer was still active.
        self.agent_navigation
            .get(&thread_id)
            .is_none_or(|entry| !entry.is_closed)
            && self
                .thread_event_channels
                .get(&thread_id)
                .is_none_or(|channel| channel.attachment() == ThreadEventAttachment::ReplayOnly)
    }

    pub(super) fn reset_for_thread_switch(&mut self, tui: &mut tui::Tui) -> Result<()> {
        self.reset_transcript_state_after_clear();
        tui.clear_pending_history_lines();
        Self::clear_terminal_for_thread_switch(&mut tui.terminal)?;
        Ok(())
    }

    pub(super) fn clear_terminal_for_thread_switch<B>(
        terminal: &mut crate::custom_terminal::Terminal<B>,
    ) -> Result<()>
    where
        B: Backend + Write,
    {
        terminal.clear_scrollback_and_visible_screen_ansi()?;
        let mut area = terminal.viewport_area;
        if area.y > 0 {
            area.y = 0;
            terminal.set_viewport_area(area);
        }
        Ok(())
    }

    pub(super) fn reset_thread_event_state(&mut self) {
        // The channels and navigation entries being discarded belong to the previous session view.
        // Late events for any of those ids must not recreate detached channels while the replacement
        // primary is being installed. A child event can arrive after the primary event that caused
        // the reset, so track the complete detached set.
        let retired_thread_ids: Vec<ThreadId> = self
            .thread_event_channels
            .keys()
            .copied()
            .chain(self.agent_navigation.tracked_thread_ids())
            .collect();
        for thread_id in retired_thread_ids {
            self.retire_thread_id(thread_id);
        }
        // The widget can still be the only owner visible during an interrupted transition. Include
        // it even when the primary field was already cleared so its delayed events cannot recreate
        // the previous session as a fresh local thread.
        if let Some(thread_id) = self.chat_widget.thread_id() {
            self.retire_thread_id(thread_id);
        }
        // A replacement can be requested after the new session id has been learned but before
        // `enqueue_primary_thread_session*` installs it as the active primary. Treat that pending
        // destination as detached too; otherwise a late event for the abandoned startup can create
        // a fresh channel during the next transition.
        if let Some(pending_primary_thread_id) = self.pending_primary_thread_id {
            self.retire_thread_id(pending_primary_thread_id);
        }
        if let Some(primary_thread_id) = self.primary_thread_id {
            self.retire_thread_id(primary_thread_id);
        }
        let detached_side_and_listener_ids: Vec<ThreadId> = self
            .side_threads
            .keys()
            .copied()
            .chain(self.thread_event_listener_tasks.keys().copied())
            .collect();
        for thread_id in detached_side_and_listener_ids {
            self.retire_thread_id(thread_id);
        }
        self.abort_all_thread_event_listeners();
        self.thread_event_channels.clear();
        self.agent_navigation.clear();
        self.side_threads.clear();
        self.active_thread_id = None;
        self.active_thread_rx = None;
        self.primary_thread_id = None;
        self.pending_primary_thread_id = None;
        self.last_subagent_backfill_attempt = None;
        self.primary_session_configured = None;
        self.pending_primary_events.clear();
        self.carried_navigation_thread_ids.clear();
        self.pending_app_server_requests.clear();
        self.pending_startup_thread_start = false;
        self.chat_widget.set_pending_thread_approvals(Vec::new());
        self.sync_active_agent_label();
    }

    pub(super) fn reset_thread_event_state_preserving_navigation(
        &mut self,
        previous_navigation: AgentNavigationState,
    ) {
        self.reset_thread_event_state();
        self.agent_navigation = previous_navigation;
        self.sync_active_agent_label();
    }

    pub(super) fn reconcile_agent_navigation_after_backfill(
        &mut self,
        backfill: &LoadedSubagentBackfill,
    ) {
        // A successful descendant read is enough to make a previously retired child eligible for
        // routing again. This also handles a resumed primary whose children were retired during a
        // prior session replacement; do it even when another page/read failed.
        for thread_id in &backfill.discovered_thread_ids {
            self.unretire_thread_id(*thread_id);
        }
        // Never prune from a partial view. A list/read failure means the absence of an entry is
        // unknown, not evidence that the persisted subagent was deleted.
        if !backfill.completed {
            return;
        }
        let Some(primary_thread_id) = self.primary_thread_id else {
            return;
        };
        let mut retained_thread_ids = backfill.discovered_thread_ids.clone();
        retained_thread_ids.insert(primary_thread_id);
        retained_thread_ids.extend(self.thread_event_channels.keys().copied());
        retained_thread_ids.extend(self.side_threads.keys().copied());
        if let Some(active_thread_id) = self.active_thread_id {
            retained_thread_ids.insert(active_thread_id);
        }
        // Entries observed after the reset are newer local evidence and must survive even when
        // the complete backend snapshot was taken before they were persisted. Only carried entries
        // are eligible for removal here.
        for thread_id in self.agent_navigation.tracked_thread_ids() {
            if !self.carried_navigation_thread_ids.contains(&thread_id) {
                retained_thread_ids.insert(thread_id);
            }
        }
        let removed_carried_thread_ids: Vec<ThreadId> = self
            .agent_navigation
            .tracked_thread_ids()
            .into_iter()
            .filter(|thread_id| {
                self.carried_navigation_thread_ids.contains(thread_id)
                    && !retained_thread_ids.contains(thread_id)
            })
            .collect();
        self.agent_navigation.retain_threads(&retained_thread_ids);
        for thread_id in removed_carried_thread_ids {
            // A complete reconciliation is the first point at which absence is authoritative. Keep
            // a tombstone for entries removed here so a delayed event cannot recreate a stale child
            // channel after the picker has pruned it.
            self.retire_thread_id(thread_id);
        }
        self.carried_navigation_thread_ids.clear();
        self.sync_active_agent_label();
    }

    pub(super) async fn handle_startup_thread_started(
        &mut self,
        app_server: &mut AppServerSession,
        result: Result<AppServerStartedThread, String>,
    ) -> Result<()> {
        if !self.pending_startup_thread_start {
            if let Ok(started) = result {
                let thread_id = started.session.thread_id;
                if let Err(err) = app_server.thread_unsubscribe(thread_id).await {
                    tracing::warn!(
                        thread_id = %thread_id,
                        "failed to unsubscribe stale startup thread: {err}"
                    );
                }
                self.discard_thread_local_state(thread_id).await;
            }
            return Ok(());
        }

        self.pending_startup_thread_start = false;
        self.chat_widget
            .set_queue_submissions_until_session_configured(/*queue*/ false);
        match result {
            Ok(started) => {
                // The startup response is the first point at which the destination primary id is
                // known. Mark it before the async session enqueue so events arriving in this window
                // are buffered as primary events and replayed after the session is configured.
                self.pending_primary_thread_id = Some(started.session.thread_id);
                self.set_primary_thread_parent_owned(
                    started.session.thread_id,
                    started.blocks_direct_input,
                );
                self.enqueue_primary_thread_session(started.session, started.turns)
                    .await?;
                self.chat_widget.maybe_send_next_queued_input();
            }
            Err(err) => {
                return Err(color_eyre::eyre::eyre!(
                    "Failed to start a fresh session through the app server: {err}"
                ));
            }
        }
        Ok(())
    }

    pub(super) async fn start_fresh_session_with_summary_hint(
        &mut self,
        tui: &mut tui::Tui,
        app_server: &mut AppServerSession,
        session_start_source: Option<ThreadStartSource>,
        initial_user_message: Option<crate::chatwidget::UserMessage>,
        new_thread_name: Option<String>,
    ) {
        // Start a fresh in-memory session while preserving resumability via persisted rollout
        // history. If an initial message is provided, `enqueue_primary_thread_session` suppresses it
        // until the new session is configured and any replayed turns have been rendered.
        self.refresh_in_memory_config_from_disk_best_effort("starting a new thread")
            .await;
        let model = self.chat_widget.current_model().to_string();
        let mut config = self.fresh_session_config();
        apply_managed_new_thread_defaults(
            &mut config,
            app_server.managed_new_thread_defaults(),
            &self.cli_kv_overrides,
            &self.harness_overrides,
        );
        let summary = session_summary(
            self.chat_widget.token_usage(),
            self.chat_widget.thread_id(),
            self.chat_widget.thread_name(),
            self.chat_widget.rollout_path().as_deref(),
        );
        self.shutdown_current_thread(app_server).await;
        let tracked_thread_ids: Vec<ThreadId> =
            self.thread_event_channels.keys().copied().collect();
        for thread_id in tracked_thread_ids {
            if let Err(err) = app_server.thread_unsubscribe(thread_id).await {
                tracing::warn!("failed to unsubscribe tracked thread {thread_id}: {err}");
            }
        }
        self.config = config.clone();
        match app_server
            .start_thread_with_session_start_source(&config, session_start_source)
            .await
        {
            Ok(mut started) => {
                let name_error = if let Some(name) = new_thread_name {
                    match app_server
                        .thread_set_name(started.session.thread_id, name.clone())
                        .await
                    {
                        Ok(()) => {
                            started.session.thread_name = Some(name);
                            None
                        }
                        Err(err) => Some(format!("Failed to name the new session: {err}")),
                    }
                } else {
                    None
                };
                if let Err(err) = self
                    .replace_chat_widget_with_app_server_thread(
                        tui,
                        started,
                        ThreadAttachPresentation::SessionLineage,
                        initial_user_message,
                    )
                    .await
                {
                    self.chat_widget.add_error_message(format!(
                        "Failed to attach to fresh app-server thread: {err}"
                    ));
                } else {
                    if let Some(err) = name_error {
                        self.chat_widget.add_error_message(err);
                    }
                    if let Some(summary) = summary {
                        let mut lines: Vec<Line<'static>> = Vec::new();
                        if let Some(usage_line) = summary.usage_line {
                            lines.push(usage_line.into());
                        }
                        if let Some(command) = summary.resume_hint {
                            let spans =
                                vec!["To continue this session, run ".into(), command.cyan()];
                            lines.push(spans.into());
                        }
                        self.chat_widget.add_plain_history_lines(lines);
                    }
                }
            }
            Err(err) => {
                self.chat_widget.add_error_message(format!(
                    "Failed to start a fresh session through the app server: {err}"
                ));
                self.config.model = Some(model);
            }
        }
        tui.frame_requester().schedule_frame();
    }

    pub(super) async fn replace_chat_widget_with_app_server_thread(
        &mut self,
        tui: &mut tui::Tui,
        started: AppServerStartedThread,
        presentation: ThreadAttachPresentation,
        initial_user_message: Option<crate::chatwidget::UserMessage>,
    ) -> Result<()> {
        self.replace_chat_widget_with_app_server_thread_inner(
            tui,
            started,
            presentation,
            initial_user_message,
            /*preserve_navigation*/ false,
        )
        .await
    }

    pub(super) async fn replace_chat_widget_with_app_server_thread_preserving_navigation(
        &mut self,
        tui: &mut tui::Tui,
        started: AppServerStartedThread,
        presentation: ThreadAttachPresentation,
        initial_user_message: Option<crate::chatwidget::UserMessage>,
    ) -> Result<()> {
        self.replace_chat_widget_with_app_server_thread_inner(
            tui,
            started,
            presentation,
            initial_user_message,
            /*preserve_navigation*/ true,
        )
        .await
    }

    async fn replace_chat_widget_with_app_server_thread_inner(
        &mut self,
        tui: &mut tui::Tui,
        started: AppServerStartedThread,
        presentation: ThreadAttachPresentation,
        initial_user_message: Option<crate::chatwidget::UserMessage>,
        preserve_navigation: bool,
    ) -> Result<()> {
        // Initial messages are for freshly attached primary threads only. Thread switches and
        // resume/fork flows pass `None` so they cannot replay old history and then auto-submit a new
        // user turn by accident.
        let previous_primary_thread_id = self.primary_thread_id;
        let previous_thread_ids: HashSet<ThreadId> = self
            .agent_navigation
            .tracked_thread_ids()
            .into_iter()
            .chain(self.thread_event_channels.keys().copied())
            .collect();
        // Only carry navigation when the requested resume really returns to the current primary.
        // Computing this set from the raw flag before the identity check would leave stale ids in
        // carried_navigation_thread_ids after a resume to a different session.
        let preserve_navigation =
            preserve_navigation && previous_primary_thread_id == Some(started.session.thread_id);
        let carried_navigation_thread_ids = if preserve_navigation {
            self.agent_navigation
                .tracked_thread_ids()
                .into_iter()
                .filter(|thread_id| Some(*thread_id) != previous_primary_thread_id)
                .collect()
        } else {
            HashSet::new()
        };
        // Cached descendants belong to the old primary only when the resume returns to that same
        // primary (for example, returning from a child transcript to its parent). Do not carry a
        // different session's agent rows into a new primary while its own backfill is running.
        if preserve_navigation {
            // Keep the primary entry in its original position. Removing it before reset and
            // re-inserting it during attach moves the main row behind its children, which makes
            // picker traversal and the active-agent label appear to switch threads during resume.
            let previous_navigation = self.agent_navigation.clone();
            self.reset_thread_event_state_preserving_navigation(previous_navigation);
        } else {
            self.reset_thread_event_state();
        }
        self.carried_navigation_thread_ids = carried_navigation_thread_ids;
        // Resuming the same primary keeps the old child ids in the same session lineage. Other
        // replacements keep the detached ids retired so delayed events from the old session cannot
        // create stale agent rows.
        if previous_primary_thread_id == Some(started.session.thread_id) {
            for thread_id in previous_thread_ids {
                self.unretire_thread_id(thread_id);
            }
        }
        self.pending_primary_thread_id = Some(started.session.thread_id);
        let init = self.chatwidget_init_for_forked_or_resumed_thread(
            tui,
            self.config.clone(),
            initial_user_message,
        );
        self.replace_chat_widget(ChatWidget::new_with_app_event(init));
        self.set_primary_thread_parent_owned(
            started.session.thread_id,
            started.blocks_direct_input,
        );
        self.enqueue_primary_thread_session_with_presentation(
            started.session,
            started.turns,
            presentation,
        )
        .await?;
        Ok(())
    }

    /// Fetches persisted and loaded descendants from the app server and registers them in the
    /// navigation cache and chat widget metadata.
    ///
    /// Called when opening the `/agent` picker and after resuming a thread so that the picker and
    /// keyboard navigation are pre-populated even if the TUI did not witness the original spawn
    /// events. Fresh and forked threads cannot have pre-existing descendants.
    ///
    /// Persisted descendants are fetched through paginated `thread/list` calls. The loaded-thread
    /// list is merged in to cover threads that have spawned but have not reached persistent
    /// indexing yet. The combined spawn tree is walked by
    /// `find_loaded_subagent_threads_for_primary`.
    pub(super) async fn backfill_loaded_subagent_threads(
        &mut self,
        app_server: &mut AppServerSession,
    ) -> LoadedSubagentBackfill {
        let Some(primary_thread_id) = self.primary_thread_id else {
            return LoadedSubagentBackfill::default();
        };

        let mut threads_by_id = HashMap::new();
        let mut cursor = None;
        let mut had_read_error = false;
        loop {
            let response = match app_server
                .thread_list(ThreadListParams {
                    cursor,
                    limit: Some(SUBAGENT_BACKFILL_PAGE_SIZE),
                    sort_key: Some(ThreadSortKey::CreatedAt),
                    sort_direction: Some(SortDirection::Asc),
                    model_providers: None,
                    source_kinds: Some(vec![ThreadSourceKind::SubAgentThreadSpawn]),
                    archived: Some(false),
                    is_pinned: None,
                    cwd: None,
                    use_state_db_only: false,
                    search_term: None,
                    parent_thread_id: None,
                    ancestor_thread_id: Some(primary_thread_id.to_string()),
                })
                .await
            {
                Ok(response) => response,
                Err(err) => {
                    had_read_error = true;
                    tracing::warn!(%err, "failed to list persisted threads for subagent backfill");
                    break;
                }
            };

            for thread in response.data {
                threads_by_id.insert(thread.id.clone(), thread);
            }
            let Some(next_cursor) = response.next_cursor else {
                break;
            };
            cursor = Some(next_cursor);
        }

        let loaded_thread_ids = match app_server
            .thread_loaded_list(ThreadLoadedListParams {
                cursor: None,
                limit: None,
            })
            .await
        {
            Ok(response) => response.data,
            Err(err) => {
                had_read_error = true;
                tracing::warn!(%err, "failed to list loaded threads for subagent backfill");
                Vec::new()
            }
        };

        for thread_id in loaded_thread_ids {
            let Ok(thread_id) = ThreadId::from_string(&thread_id) else {
                tracing::warn!("ignoring loaded thread with invalid id during subagent backfill");
                continue;
            };

            if thread_id == primary_thread_id {
                continue;
            }

            match app_server
                .thread_read(thread_id, /*include_turns*/ false)
                .await
            {
                Ok(thread) => {
                    threads_by_id.insert(thread.id.clone(), thread);
                }
                Err(err) => {
                    had_read_error = true;
                    tracing::warn!(thread_id = %thread_id, %err, "failed to read loaded thread");
                }
            }
        }

        let mut refreshed_thread_ids = HashSet::new();
        let mut discovered_thread_ids = HashSet::new();
        for thread in find_loaded_subagent_threads_for_primary(
            threads_by_id.into_values().collect(),
            primary_thread_id,
        ) {
            discovered_thread_ids.insert(thread.thread_id);
            let agent_path = thread.agent_path;
            let has_live_channel = self
                .thread_event_channels
                .get(&thread.thread_id)
                .is_some_and(|channel| channel.attachment() == ThreadEventAttachment::Live);
            let is_closed = !has_live_channel && thread.is_closed;
            self.agent_navigation
                .set_parent_owned(thread.thread_id, thread.blocks_direct_input);
            self.upsert_agent_picker_thread(
                thread.thread_id,
                thread.agent_nickname,
                thread.agent_role,
                is_closed,
            );
            self.agent_navigation
                .set_agent_path(thread.thread_id, agent_path);
            // A live channel can have an empty store after a successful spawn. Only apply server
            // status for channels that would otherwise need another liveness read.
            if !has_live_channel {
                if thread.is_running {
                    self.agent_navigation.mark_running(thread.thread_id);
                } else {
                    self.agent_navigation
                        .set_running(thread.thread_id, /*is_running*/ false);
                }
                refreshed_thread_ids.insert(thread.thread_id);
            }
        }
        for thread_id in &discovered_thread_ids {
            self.unretire_thread_id(*thread_id);
        }
        self.sync_active_agent_label();

        LoadedSubagentBackfill {
            completed: !had_read_error,
            refreshed_thread_ids,
            discovered_thread_ids,
        }
    }

    /// Returns the adjacent thread id for keyboard navigation, backfilling from the server if the
    /// local cache has no neighbor.
    ///
    /// Tries the fast path first: ask `AgentNavigationState` directly. If it returns `None` (no
    /// adjacent entry exists, typically because the cache was never populated with remote
    /// subagents), performs a full `backfill_loaded_subagent_threads` and retries. This ensures the
    /// first next/previous keypress in a resumed remote session discovers subagents on demand
    /// without requiring the user to wait for a proactive fetch.
    pub(super) async fn adjacent_thread_id_with_backfill(
        &mut self,
        app_server: &mut AppServerSession,
        direction: AgentNavigationDirection,
    ) -> Option<ThreadId> {
        let current_thread = self.current_displayed_thread_id();
        if let Some(thread_id) = self
            .agent_navigation
            .adjacent_thread_id(current_thread, direction)
        {
            return Some(thread_id);
        }

        let primary_thread_id = self.primary_thread_id?;
        if self.last_subagent_backfill_attempt == Some(primary_thread_id) {
            return None;
        }

        let backfill = self.backfill_loaded_subagent_threads(app_server).await;
        self.reconcile_agent_navigation_after_backfill(&backfill);
        if backfill.completed {
            self.last_subagent_backfill_attempt = Some(primary_thread_id);
        }
        self.agent_navigation
            .adjacent_thread_id(self.current_displayed_thread_id(), direction)
    }

    pub(super) fn fresh_session_config(&self) -> Config {
        let mut config = self.config.clone();
        config.service_tier = self.chat_widget.configured_service_tier();
        config
    }
    pub(super) async fn resume_target_session(
        &mut self,
        tui: &mut tui::Tui,
        app_server: &mut AppServerSession,
        target_session: SessionTarget,
    ) -> Result<AppRunControl> {
        if self.ignore_same_thread_resume(&target_session) {
            tui.frame_requester().schedule_frame();
            return Ok(AppRunControl::Continue);
        }

        self.refresh_in_memory_config_from_disk_best_effort("resuming a thread")
            .await;
        let cwd_override = self
            .harness_overrides
            .cwd
            .as_deref()
            .or_else(|| app_server.remote_cwd_override());
        let resume_cwd_mode = crate::session_resume::effective_resume_cwd_mode(
            self.config.tui_resume_cwd,
            cwd_override,
        );
        let remembered_current_cwd = cwd_override.unwrap_or(self.launch_cwd.as_path());
        let current_cwd = if matches!(resume_cwd_mode, Some(ResumeCwdMode::Current)) {
            remembered_current_cwd.to_path_buf()
        } else {
            self.config.cwd.to_path_buf()
        };
        let uses_remote_workspace_or_environment = crate::uses_remote_workspace_or_environment(
            &self.app_server_target,
            &self.environment_manager,
        );
        if uses_remote_workspace_or_environment
            && self.harness_overrides.cwd.is_none()
            && app_server.remote_cwd_override().is_none()
            && matches!(resume_cwd_mode, Some(ResumeCwdMode::Current))
        {
            self.chat_widget.add_error_message(
                "`tui.resume_cwd = \"current\"` requires `--cd` when using a remote workspace"
                    .to_string(),
            );
            return Ok(AppRunControl::Continue);
        }
        let resume_cwd = if self.app_server_target.uses_remote_workspace() {
            current_cwd.clone()
        } else {
            let outcome = crate::session_resume::resolve_cwd_for_resume_or_fork(
                tui,
                &self.config,
                self.state_db.as_deref(),
                &target_session,
                CwdPromptAction::Resume,
                crate::session_resume::ResumeCwdContext {
                    current_cwd: &current_cwd,
                    remembered_current_cwd,
                    allow_remember_current: !uses_remote_workspace_or_environment
                        || cwd_override.is_some(),
                    mode: resume_cwd_mode,
                },
            )
            .await;
            match outcome {
                Err(err) => {
                    self.chat_widget.add_error_message(format!(
                        "Failed to determine working directory for resume: {err}"
                    ));
                    return Ok(AppRunControl::Continue);
                }
                Ok(crate::session_resume::ResolveCwdOutcome::Continue(Some(cwd))) => cwd,
                Ok(crate::session_resume::ResolveCwdOutcome::Continue(None)) => current_cwd.clone(),
                Ok(crate::session_resume::ResolveCwdOutcome::Exit) => {
                    return Ok(AppRunControl::Exit(ExitReason::UserRequested));
                }
            }
        };

        let (config_current_cwd, config_resume_cwd) =
            if self.app_server_target.uses_remote_workspace() {
                let local_config_cwd = self.config.cwd.to_path_buf();
                (local_config_cwd.clone(), local_config_cwd)
            } else {
                (current_cwd, resume_cwd)
            };
        let mut resume_config = match self
            .rebuild_config_for_resume_or_fallback(&config_current_cwd, config_resume_cwd)
            .await
        {
            Ok(cfg) => cfg,
            Err(err) => {
                self.chat_widget.add_error_message(format!(
                    "Failed to rebuild configuration for resume: {err}"
                ));
                return Ok(AppRunControl::Continue);
            }
        };
        self.apply_runtime_policy_overrides(&mut resume_config);

        let summary = session_summary(
            self.chat_widget.token_usage(),
            self.chat_widget.thread_id(),
            self.chat_widget.thread_name(),
            self.chat_widget.rollout_path().as_deref(),
        );
        match app_server
            .resume_thread(
                resume_config.clone(),
                target_session.thread_id,
                self.resume_model_settings(),
            )
            .await
        {
            Ok(resumed) => {
                let resumed_thread_id = resumed.session.thread_id;
                self.shutdown_current_thread(app_server).await;
                self.config = resume_config;
                tui.set_notification_settings(
                    self.config.tui_notifications.method,
                    self.config.tui_notifications.condition,
                );
                self.file_search
                    .update_search_dir(self.config.cwd.to_path_buf());
                match self
                    .replace_chat_widget_with_app_server_thread_preserving_navigation(
                        tui,
                        resumed,
                        ThreadAttachPresentation::SessionLineage,
                        /*initial_user_message*/ None,
                    )
                    .await
                {
                    Ok(()) => {
                        let backfill = self.backfill_loaded_subagent_threads(app_server).await;
                        self.reconcile_agent_navigation_after_backfill(&backfill);
                        if !backfill.completed {
                            tracing::warn!(
                                "subagent backfill was incomplete after resume; retaining cached navigation state"
                            );
                        }
                        if let Some(summary) = summary {
                            let mut lines: Vec<Line<'static>> = Vec::new();
                            if let Some(usage_line) = summary.usage_line {
                                lines.push(usage_line.into());
                            }
                            if let Some(command) = summary.resume_hint {
                                let spans =
                                    vec!["To continue this session, run ".into(), command.cyan()];
                                lines.push(spans.into());
                            }
                            self.chat_widget.add_plain_history_lines(lines);
                        }
                        self.maybe_prompt_resume_paused_goal_after_resume(
                            app_server,
                            resumed_thread_id,
                        )
                        .await;
                    }
                    Err(err) => {
                        self.chat_widget.add_error_message(format!(
                            "Failed to attach to resumed app-server thread: {err}"
                        ));
                    }
                }
            }
            Err(err) => {
                let path_display = target_session.display_label();
                self.chat_widget.add_error_message(format!(
                    "Failed to resume session from {path_display}: {err}"
                ));
            }
        }

        Ok(AppRunControl::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::ThreadId;
    use std::collections::HashSet;

    #[tokio::test]
    async fn partial_backfill_reconciliation_preserves_cached_agent_navigation() {
        let mut app = crate::app::test_support::make_test_app().await;
        let primary_thread_id = ThreadId::new();
        let child_thread_id = ThreadId::new();
        let stale_thread_id = ThreadId::new();

        app.primary_thread_id = Some(primary_thread_id);
        app.agent_navigation.upsert(
            primary_thread_id,
            /*agent_nickname*/ None,
            /*agent_role*/ None,
            /*is_closed*/ false,
        );
        app.agent_navigation.upsert(
            child_thread_id,
            Some("worker".to_string()),
            Some("child".to_string()),
            /*is_closed*/ false,
        );
        app.agent_navigation.upsert(
            stale_thread_id,
            Some("stale".to_string()),
            /*agent_role*/ None,
            /*is_closed*/ false,
        );
        app.carried_navigation_thread_ids = HashSet::from([child_thread_id, stale_thread_id]);

        app.reconcile_agent_navigation_after_backfill(&LoadedSubagentBackfill {
            completed: false,
            refreshed_thread_ids: HashSet::new(),
            discovered_thread_ids: HashSet::new(),
        });
        assert!(app.agent_navigation.get(&child_thread_id).is_some());
        assert!(app.agent_navigation.get(&stale_thread_id).is_some());

        app.reconcile_agent_navigation_after_backfill(&LoadedSubagentBackfill {
            completed: true,
            refreshed_thread_ids: HashSet::new(),
            discovered_thread_ids: HashSet::from([child_thread_id]),
        });
        assert!(app.agent_navigation.get(&primary_thread_id).is_some());
        assert!(app.agent_navigation.get(&child_thread_id).is_some());
        assert!(app.agent_navigation.get(&stale_thread_id).is_none());
    }

    #[test]
    fn terminal_thread_read_error_detection_matches_not_loaded_errors() {
        let err = color_eyre::eyre::eyre!(
            "thread/read failed during TUI session lookup: thread/read failed: thread not loaded: thr_123"
        );

        assert!(!App::is_terminal_thread_read_error(&err));
    }

    #[test]
    fn terminal_thread_read_error_detection_ignores_transient_failures() {
        let err = color_eyre::eyre::eyre!(
            "thread/read failed during TUI session lookup: thread/read transport error: broken pipe"
        );

        assert!(!App::is_terminal_thread_read_error(&err));
    }

    #[test]
    fn terminal_thread_read_error_detection_matches_only_confirmed_missing_threads() {
        for message in [
            "thread/read failed: thread not found: thr_123",
            "thread/read failed: thread does not exist: thr_123",
            "thread/read failed: no rollout found for thread id thr_123",
        ] {
            let err = color_eyre::eyre::eyre!(message);
            assert!(App::is_terminal_thread_read_error(&err), "{message}");
        }
        let not_loaded = color_eyre::eyre::eyre!("thread/read failed: thread not loaded: thr_123");
        assert!(!App::is_terminal_thread_read_error(&not_loaded));
    }

    #[test]
    fn closed_state_for_thread_read_error_preserves_live_state_without_cache_on_transient_error() {
        let err = color_eyre::eyre::eyre!(
            "thread/read failed during TUI session lookup: thread/read transport error: broken pipe"
        );

        assert!(!App::closed_state_for_thread_read_error(
            &err, /*existing_is_closed*/ None
        ));
    }

    #[test]
    fn closed_state_for_thread_read_error_marks_terminal_uncached_threads_closed() {
        let err = color_eyre::eyre::eyre!(
            "thread/read failed during TUI session lookup: thread/read failed: thread not found: thr_123"
        );

        assert!(App::closed_state_for_thread_read_error(
            &err, /*existing_is_closed*/ None
        ));
    }

    #[test]
    fn thread_not_loaded_keeps_existing_open_state() {
        let err = color_eyre::eyre::eyre!(
            "thread/read failed during TUI session lookup: thread/read failed: thread not loaded: thr_123"
        );

        assert!(!App::closed_state_for_thread_read_error(&err, Some(false)));
    }

    #[test]
    fn include_turns_fallback_detection_handles_unmaterialized_and_ephemeral_threads() {
        let unmaterialized = color_eyre::eyre::eyre!(
            "thread/read failed during TUI session lookup: thread/read failed: thread thr_123 is not materialized yet; includeTurns is unavailable before first user message"
        );
        let ephemeral = color_eyre::eyre::eyre!(
            "thread/read failed during TUI session lookup: thread/read failed: ephemeral threads do not support includeTurns"
        );

        assert!(App::can_fallback_from_include_turns_error(&unmaterialized));
        assert!(App::can_fallback_from_include_turns_error(&ephemeral));
    }
}
