use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::time::{Duration, Instant};
use qdrant_client::qdrant::{SearchParams, SearchPoints};
use qdrant_client::Qdrant;
use tracing::{debug, info};

use crate::embeddings::EmbeddingEngine;

/// Structure representing a standardized test query
#[derive(Debug, Clone)]
pub struct BenchmarkQuery {
    pub text: String,
    pub description: String,
}

/// Latency metrics extracted from execution runs
#[derive(Debug, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub mean_ms: f64,
}

/// Final performance report for a single query
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub query: String,
    pub top_k: usize,
    pub iterations: usize,
    pub recall_at_k: f64,
    pub latency: LatencyMetrics,
}

pub struct BenchmarkSuite {
    client: Qdrant,
    collection_name: String,
}

impl BenchmarkSuite {
    pub fn new(client: Qdrant, collection_name: &str) -> Self {
        Self {
            client,
            collection_name: collection_name.to_string(),
        }
    }

    /// Retrieves Ground Truth Point IDs using exact brute-force search
    async fn get_ground_truth(
        &self,
        vector: &[f32],
        k: usize,
    ) -> Result<HashSet<u64>> {
        let search_res = self
            .client
            .search_points(SearchPoints {
                collection_name: self.collection_name.clone(),
                vector: vector.to_vec(),
                limit: k as u64,
                params: Some(SearchParams {
                    exact: Some(true), // Forces exact sequential search in Qdrant
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await?;

        let ground_truth_ids = search_res
            .result
            .into_iter()
            .map(|p| match p.id {
                Some(qdrant_client::qdrant::PointId {
                         point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(id)),
                     }) => id,
                _ => 0,
            })
            .collect();

        Ok(ground_truth_ids)
    }

    /// Evaluates p50/p95 latencies and calculates Recall@K comparing HNSW vs Exact search
    pub async fn evaluate_query(
        &self,
        query: &BenchmarkQuery,
        embed_engine: &mut EmbeddingEngine,
        top_k: usize,
        iterations: usize,
    ) -> Result<BenchmarkReport> {
        info!("Evaluating query: '{}'", query.text);

        // 1. Generate embedding vector for the target query
        let query_vectors = embed_engine.embed_chunks(&[query.text.clone()])?;
        let query_vector = &query_vectors[0];

        // 2. Fetch ground truth point IDs for Recall@K evaluation
        let ground_truth_ids = self.get_ground_truth(query_vector, top_k).await?;

        // 3. Run iterations to measure HNSW index query latencies
        let mut durations: Vec<Duration> = Vec::with_capacity(iterations);
        let mut last_retrieved_ids: HashSet<u64> = HashSet::new();

        for _ in 0..iterations {
            let start = Instant::now();

            let response = self
                .client
                .search_points(SearchPoints {
                    collection_name: self.collection_name.clone(),
                    vector: query_vector.clone(),
                    limit: top_k as u64,
                    ..Default::default()
                })
                .await?;

            let elapsed = start.elapsed();
            durations.push(elapsed);

            // Store retrieved IDs from the latest run to compute Recall
            last_retrieved_ids = response
                .result
                .into_iter()
                .map(|p| match p.id {
                    Some(qdrant_client::qdrant::PointId {
                             point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(id)),
                         }) => id,
                    _ => 0,
                })
                .collect();
        }

        // 4. Calculate latency metrics
        durations.sort();
        let min_ms = durations.first().unwrap().as_secs_f64() * 1000.0;
        let max_ms = durations.last().unwrap().as_secs_f64() * 1000.0;
        let p50_ms = durations[(iterations as f64 * 0.50) as usize].as_secs_f64() * 1000.0;
        let p95_ms = durations[(iterations as f64 * 0.95) as usize].as_secs_f64() * 1000.0;
        let sum_ms: f64 = durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
        let mean_ms = sum_ms / iterations as f64;

        // 5. Compute Recall@K: |Retrieved ∩ GroundTruth| / K
        let intersection_count = last_retrieved_ids.intersection(&ground_truth_ids).count();
        let recall_at_k = intersection_count as f64 / top_k as f64;

        debug!(
            "Query evaluation complete. p50: {:.2}ms, p95: {:.2}ms, Recall@{}: {:.2}",
            p50_ms, p95_ms, top_k, recall_at_k
        );

        Ok(BenchmarkReport {
            query: query.text.clone(),
            top_k,
            iterations,
            recall_at_k,
            latency: LatencyMetrics {
                min_ms,
                max_ms,
                p50_ms,
                p95_ms,
                mean_ms,
            },
        })
    }

    /// Exports benchmark results to a formatted JSON file on disk
    pub fn export_results(&self, reports: &[BenchmarkReport], output_path: &str) -> Result<()> {
        let file = File::create(output_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, reports)?;
        info!("Benchmark results successfully exported to '{}'", output_path);
        Ok(())
    }
}