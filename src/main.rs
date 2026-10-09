mod ingestion; 

use anyhow::Result;
use qdrant_client::Qdrant;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Inicializar sistema de logs
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Iniciando VectorLab Benchmark Suite...");

    // 2. Comprobar conexión con Qdrant
    let qdrant_url = "http://localhost:6334";
    let client = Qdrant::from_url(qdrant_url).build()?;
    let health = client.health_check().await?;
    info!("Conexión exitosa con Qdrant. Versión: {}", health.title);

    // 3. Fase 1: Ingesta y Chunking
    info!("Cargando dataset local...");
    let dataset_path = "datasets/sample.json";
    let documents = ingestion::load_dataset(dataset_path)?;
    info!("Se han cargado {} documentos.", documents.len());

    let chunk_size = 15; // Palabras por fragmento
    let overlap = 5;     // Palabras que se solapan con el siguiente fragmento
    let mut all_chunks = Vec::new();

    for doc in documents {
        let chunks = ingestion::chunk_text(&doc.id, &doc.content, chunk_size, overlap);
        all_chunks.extend(chunks);
    }

    info!("El texto se ha dividido en un total de {} fragmentos.", all_chunks.len());

    // Imprimimos el primer fragmento para verificar visualmente
    if let Some(first) = all_chunks.first() {
        info!("Ejemplo de fragmento ({}): '{}'", first.chunk_id, first.text);
    }

    Ok(())
}