# VectorLab: Semantic Search & Vector Database Benchmarking Suite

VectorLab is an experimental benchmarking environment built in Rust to analyze how embedding models, index parameters, and chunking strategies impact retrieval quality, latency, and memory consumption.

## 🔬 Research Question
> *Which combination of embedding model, chunk size, and vector search strategy (Exact vs. HNSW) yields the optimal trade-off between retrieval quality (Recall@K, nDCG), latency (p50/p95), and RAM footprint?*

## 🛠️ Architecture & Tech Stack
- **Core Engine:** Rust (Tokio async runtime)
- **Embeddings:** FastEmbed (Local ONNX execution using `bge-small-en-v1.5`)
- **Vector Database:** Qdrant (gRPC integration)
- **Evaluation Framework:** Python (Polars & Seaborn for data visualization)
- **Containerization:** Docker Compose

## 📊 Evaluated Metrics
- **Recall@K:** Proportion of ground-truth relevant documents retrieved in the top K results.
- **nDCG@K:** Normalized Discounted Cumulative Gain assessing ranking quality.
- **Latency (p50 / p95):** Query response duration distribution under search workloads.
- **Resource Usage:** Index size and memory footprint.

## 🚀 Quickstart

### Prerequisites
- Rust (1.75+)
- Docker & Docker Compose

### 1. Launch Vector DB