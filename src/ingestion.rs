use anyhow::Result;
use serde::Deserialize;
use std::fs;

// Estructura que representa el documento original del JSON
#[derive(Deserialize, Debug)]
pub struct RawDocument {
    pub id: String,
    pub content: String,
}

// Estructura para los fragmentos resultantes
#[derive(Debug)]
pub struct DocumentChunk {
    pub document_id: String,
    pub chunk_id: String,
    pub text: String,
}

/// Lee un archivo JSON y lo convierte en un vector de documentos
pub fn load_dataset(path: &str) -> Result<Vec<RawDocument>> {
    let data = fs::read_to_string(path)?;
    let docs: Vec<RawDocument> = serde_json::from_str(&data)?;
    Ok(docs)
}

/// Fragmenta un texto basándose en número de palabras, aplicando un solapamiento (overlap)
pub fn chunk_text(
    doc_id: &str,
    text: &str,
    chunk_size: usize,
    overlap: usize,
) -> Vec<DocumentChunk> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    let mut chunk_index = 0;

    // Validación de seguridad para evitar bucles infinitos
    let step = if chunk_size > overlap { chunk_size - overlap } else { 1 };

    while i < words.len() {
        let end = std::cmp::min(i + chunk_size, words.len());
        let chunk_text = words[i..end].join(" ");

        chunks.push(DocumentChunk {
            document_id: doc_id.to_string(),
            chunk_id: format!("{}_chunk_{}", doc_id, chunk_index),
            text: chunk_text,
        });

        if end == words.len() {
            break;
        }

        i += step;
        chunk_index += 1;
    }

    chunks
}