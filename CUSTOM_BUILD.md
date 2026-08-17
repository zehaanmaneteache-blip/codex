# Custom Codex Build

This branch is based on the official `rust-v0.146.1` tag and builds only the
target used by the current installation:

```text
x86_64-unknown-linux-musl
```

## Direct subagent input

The custom behavior is disabled by default. Enable it in `config.toml`:

```toml
[features.multi_agent_v2]
allow_direct_subagent_input = true
```

When enabled, the app server advertises direct-input support for multi-agent v2
subagents created through `ThreadSpawn`, and accepts direct turn and steer
requests for those threads. Other thread types and the default policy are
unchanged.

## Subagent recovery after restart

The TUI merges paginated persisted `thread/list` results with the in-memory
`thread/loaded/list` results when it rebuilds `/agent`. Descendants are selected
by `ancestor_thread_id`, so subagents from the resumed root thread remain in the
picker after Codex restarts. Selecting a persisted cold subagent resumes it with
`thread/resume`; it can then receive messages normally when direct subagent
input is enabled.

## Provider and model hot reload

Before a new `turn/start`, the TUI and embedded app server reload the effective
configuration. Changes to the selected provider, base URL, headers, configured
credentials, and provider catalog therefore apply to the next turn without a
Codex restart. An in-flight turn keeps the provider snapshot with which it
started.

Opening `/model` performs an explicit provider reload and a fresh `model/list`
request. Custom providers are queried through their standard OpenAI-compatible
`GET /models` endpoint. Responses in both Codex's extended `models` format and
the standard `{"data":[{"id":"model-name"}]}` format are accepted, so the
picker is not limited to bundled GPT model IDs.

The standard OpenAI `/models` schema only guarantees a model ID; it does not
define reasoning capability metadata. This build recognizes the optional
provider extensions `supported_reasoning_efforts`,
`supported_reasoning_levels`, or `reasoning_efforts`, plus
`default_reasoning_effort`, context-window fields, and input modalities. Values
such as `max`, `ultra`, and unknown provider-defined effort names are preserved.
When a provider omits those extensions, Codex uses bundled metadata for a known
model ID and a conservative fallback for an unknown ID; it does not guess that
the model supports a particular reasoning effort.

## GitHub Actions build

The `custom-linux-musl` workflow follows the Linux musl leg of the official
release workflow. It uses the pinned Rust and Zig versions, the repository's
musl setup script, the verified official `rusty_v8` artifacts, and the official
`bwrap` digest flow. It also runs `cargo fmt --check` before compilation.

The workflow runs on pushes to `custom/**` that affect the Rust build and can
also be started manually. It does not build macOS, Windows, ARM, DMG, npm,
Python, symbols, or standalone app-server artifacts.

Each run uploads one artifact containing:

- `codex`
- `bwrap`
- `SHA256SUMS`
- `BUILD_INFO`

The workflow does not require repository secrets and receives read-only
repository contents permission. Authentication used to push the branch remains
outside the workflow.
