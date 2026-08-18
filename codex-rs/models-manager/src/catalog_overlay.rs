use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelsResponse;
use codex_utils_path::resolve_symlink_write_paths;
use codex_utils_path::write_atomically;
use std::io;
use std::path::PathBuf;

/// Persistent metadata seed that grows as a provider discovers new model IDs.
#[derive(Debug)]
pub(crate) struct ModelCatalogOverlay {
    path: PathBuf,
}

impl ModelCatalogOverlay {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Merge provider discoveries without replacing metadata explicitly stored in the overlay.
    pub(crate) fn merge_discovered(
        &self,
        existing: &[ModelInfo],
        discovered: Vec<ModelInfo>,
    ) -> (Vec<ModelInfo>, bool) {
        let mut merged = existing.to_vec();
        let mut changed = false;
        for model in discovered {
            if merged.iter().any(|existing| existing.slug == model.slug) {
                continue;
            }
            merged.push(model);
            changed = true;
        }
        (merged, changed)
    }

    pub(crate) fn persist(&self, models: Vec<ModelInfo>) -> io::Result<()> {
        let write_path = resolve_symlink_write_paths(&self.path)?.write_path;
        let mut contents = serde_json::to_string_pretty(&ModelsResponse { models })
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
        contents.push('\n');
        write_atomically(&write_path, &contents)
    }
}
