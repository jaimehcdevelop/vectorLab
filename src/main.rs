mod benchmark;
mod embeddings;
mod ingestion;

use anyhow::Result;
use benchmark::{BenchmarkQuery, BenchmarkSuite};
use embeddings::EmbeddingEngine;
use qdrant_client::qdrant::{
    vectors_config::Config, CreateCollection, Distance, PointStruct, VectorParams, VectorsConfig,
};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use serde_json::json;
use tracing::{ info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Initialize Logging system
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting VectorLab Benchmark Suite...");

    // 2. Verify Qdrant database connection
    let qdrant_url = "http://localhost:6334";
    let client = Qdrant::from_url(qdrant_url).build()?;
    let health = client.health_check().await?;
    info!("Connected to Qdrant. Engine Version: {}", health.title);

    // 3. Phase 1: Load Structured Dataset & Perform Text Chunking
    let dataset_path = "datasets/programming-languages.json";
    let languages = ingestion::load_dataset(dataset_path)?;

    let mut all_chunks = Vec::new();
    for lang in &languages {
        let content_text = lang.to_content_text();
        let chunks = ingestion::chunk_text(&lang.name, &content_text, 20, 5);
        all_chunks.extend(chunks);
    }
    info!("Phase 1 Complete: Generated {} text chunks.", all_chunks.len());

    // 4. Phase 2: Local Vector Embeddings Execution
    let mut embed_engine = EmbeddingEngine::new()?;
    let chunk_texts: Vec<String> = all_chunks.iter().map(|c| c.text.clone()).collect();
    let vectors = embed_engine.embed_chunks(&chunk_texts)?;
    info!("Phase 2 Complete: Generated {} dense vectors.", vectors.len());

    // 5. Phase 3: Storage & Indexing in Qdrant Vector Database
    let collection_name = "programming_languages";
    let _ = client.delete_collection(collection_name).await;

    client
        .create_collection(CreateCollection {
            collection_name: collection_name.to_string(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 384,
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await?;

    let mut points = Vec::new();
    for (idx, (chunk, vector)) in all_chunks.iter().zip(vectors.into_iter()).enumerate() {
        let payload: Payload = json!({
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "word_count": chunk.word_count,
        })
            .try_into()?;

        points.push(PointStruct::new(idx as u64, vector, payload));
    }

    client
        .upsert_points(qdrant_client::qdrant::UpsertPoints {
            collection_name: collection_name.to_string(),
            points,
            ..Default::default()
        })
        .await?;

    info!("Phase 3 Complete: All vectors indexed in Qdrant.");

    // 6. Phase 4: Benchmarking & Performance Evaluation
    info!("Phase 4: Starting Automated Benchmark Suite...");

    let bench_suite = BenchmarkSuite::new(client, collection_name);

    // Standardized test queries
    let test_queries = vec![
        BenchmarkQuery {
            text: "Which object-oriented language was created for systemic applications?".to_string(),
            description: "Paradigm and use-case query".to_string(),
        },
        BenchmarkQuery {
            text: "Functional programming languages running on Java Virtual Machine".to_string(),
            description: "Query with environment and runtime constraints".to_string(),
        },
        BenchmarkQuery {
            text: "High-performance systems language with memory safety without garbage collector".to_string(),
            description: "Query targeting specific technical features".to_string(),
        },
    ];

    let mut reports = Vec::new();
    let top_k = 5;
    let iterations = 100; // Repeated measurements for stable p50/p95 calculations

    for query in &test_queries {
        let report = bench_suite
            .evaluate_query(query, &mut embed_engine, top_k, iterations)
            .await?;
        reports.push(report);
    }

    // Export final evaluation results to JSON file
    let output_file = "benchmarks/benchmark_results.json";
    bench_suite.export_results(&reports, output_file)?;

    info!("VectorLab Benchmark Suite finished successfully.");
    Ok(())
}