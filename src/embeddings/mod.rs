use anyhow::Result;
use fastembed::{TextEmbedding, TextInitOptions};
use tracing::{debug, info};

pub struct EmbeddingEngine {
    model: TextEmbedding,
}

impl EmbeddingEngine {
    /// Initializes local FastEmbed model (defaults to BGE-Small-EN-v1.5, 384 dimensions)
    pub fn new() -> Result<Self> {
        debug!("Configuring TextInitOptions for FastEmbed...");
        // Option 1: Since BGESmallENV15 is the default model,
        // we simply use the default options and enable download progress.
        let options = TextInitOptions::default().with_show_download_progress(true);

        debug!("Attempting to initialize TextEmbedding model (this may trigger a download if not cached)...");
        let model = TextEmbedding::try_new(options)?;

        // Option 2 (For the future): If you ever need to use a DIFFERENT model,
        // the model is passed in the `new` function instead of `with_model`:
        /*
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(true),
        )?;
        */

        info!("TextEmbedding model successfully loaded into memory.");
        Ok(Self { model })
    }

    /// Generates dense vector embeddings for a slice of text chunks
    /// Note: changed back to `&self` since `embed` does not require mutability in fastembed
    pub fn embed_chunks(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        debug!("Received {} text chunks for embedding generation.", texts.len());

        debug!("Invoking model.embed() processing...");
        let embeddings = self.model.embed(texts.to_vec(), None)?;

        debug!("Successfully generated embeddings for all {} chunks.", embeddings.len());
        Ok(embeddings)
    }
}