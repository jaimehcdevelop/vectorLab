use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use tracing::{debug, trace};

/// Structured input representing a programming language entry
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LanguageDoc {
    pub name: String,
    pub year: u16,
    pub creator: String,
    pub paradigm: Vec<String>,
    pub typing: String,
}

impl LanguageDoc {
    /// Converts structured JSON fields into natural language text for embedding
    pub fn to_content_text(&self) -> String {
        trace!("Converting {} data to natural language text", self.name);
        format!(
            "{} is a {} programming language created in {} by {}. Paradigms: {}.",
            self.name,
            self.typing,
            self.year,
            self.creator,
            self.paradigm.join(", ")
        )
    }
}

/// Enriched chunk structure holding text and analytical metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentChunk {
    pub document_id: String,
    pub chunk_id: String,
    pub text: String,
    pub word_count: usize,
    pub char_count: usize,
    pub chunk_index: usize,
}

/// Wrapper struct to match the root "languages" key in the JSON file
#[derive(Deserialize)]
struct DatasetWrapper {
    languages: Vec<LanguageDoc>,
}

/// Loads the language JSON dataset from disk
pub fn load_dataset(path: &str) -> Result<Vec<LanguageDoc>> {
    debug!("Reading dataset file from path: {}", path);
    let data = fs::read_to_string(path)?;

    debug!("File read successfully. Parsing JSON content using DatasetWrapper...");
    let wrapper: DatasetWrapper = serde_json::from_str(&data)?;

    debug!("Successfully parsed {} languages from JSON.", wrapper.languages.len());
    Ok(wrapper.languages)
}

/// Splits a raw text string into chunks based on word count with sliding window overlap
pub fn chunk_text(
    doc_id: &str,
    text: &str,
    chunk_size: usize,
    overlap: usize,
) -> Vec<DocumentChunk> {
    trace!("Chunking text for document ID: {}. Size: {}, Overlap: {}", doc_id, chunk_size, overlap);

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        trace!("Text is empty for document {}, returning empty chunk list.", doc_id);
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut i = 0;
    let mut chunk_index = 0;

    let step = if chunk_size > overlap { chunk_size - overlap } else { 1 };
    trace!("Calculated sliding window step size: {}", step);

    while i < words.len() {
        let end = std::cmp::min(i + chunk_size, words.len());
        let chunk_words = &words[i..end];
        let chunk_text = chunk_words.join(" ");

        trace!("Creating chunk {} (words {} to {})", chunk_index, i, end - 1);

        chunks.push(DocumentChunk {
            document_id: doc_id.to_string(),
            chunk_id: format!("{}_chunk_{}", doc_id, chunk_index),
            word_count: chunk_words.len(),
            char_count: chunk_text.chars().count(),
            chunk_index,
            text: chunk_text,
        });

        if end == words.len() {
            trace!("Reached end of words for document {}.", doc_id);
            break;
        }

        i += step;
        chunk_index += 1;
    }

    trace!("Finished chunking document {}. Total chunks: {}", doc_id, chunks.len());
    chunks
}