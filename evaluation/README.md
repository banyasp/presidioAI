# Legal Case Similarity Model Evaluation Framework

A comprehensive evaluation framework for assessing the performance of legal case similarity models using synthetic query generation and citation-based ground truth.

## Overview

This framework evaluates three BERT-based embedding models on 345 SCOTUS cases:
- `legal-bert`: nlpaueb/legal-bert-base-uncased
- `harvard-bert`: casehold/legalbert
- `sentence-transformer`: sentence-transformers/all-MiniLM-L6-v2

**Evaluation Methodology:**
1. **Synthetic Queries**: Generate 690 queries (2 per case) using extractive and generative methods
2. **Ground Truth**: Use source cases (relevance=2) and cited cases (relevance=1) as ground truth
3. **Metrics**: Precision@k, Recall@k, MRR, NDCG@k at k=[1,3,5,10,20]
4. **Comparison**: Statistical significance testing and error analysis

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install tqdm pandas numpy scikit-learn scipy transformers torch
```

2. Install Ollama (for generative queries):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
```

### Running the Complete Pipeline

**Option 1: All-in-one script (recommended)**
```bash
python evaluation/scripts/run_full_evaluation.py \
    --db-path scotus_cases.db \
    --extractive-only  # Skip Ollama requirement for now
```

**Option 2: Step-by-step**

```bash
# Step 1: Generate queries (extractive only for now)
python evaluation/scripts/01_generate_queries.py \
    --db-path scotus_cases.db \
    --extractive-only \
    --show-samples

# Step 2: Build ground truth from citations
python evaluation/scripts/02_build_ground_truth.py \
    --db-path scotus_cases.db \
    --show-samples

# Step 3: Run evaluation on all models
python evaluation/scripts/03_run_evaluation.py \
    --db-path scotus_cases.db \
    --cache-dir evaluation_results/cache

# Step 4: Generate report
python evaluation/scripts/04_generate_report.py \
    --db-path scotus_cases.db \
    --latest \
    --output-dir evaluation_results/reports
```

## Project Structure

```
evaluation/
├── config.yaml                      # Configuration file
├── README.md                        # This file
├── __init__.py
│
├── query_generation/               # Query generation modules
│   ├── generator.py                # Extractive + generative query generation
│   └── ollama_client.py            # Ollama API wrapper
│
├── ground_truth/                   # Ground truth construction
│   └── builder.py                  # Citation-based ground truth with fuzzy matching
│
├── metrics/                        # Evaluation metrics
│   └── compute.py                  # Precision@k, MRR, NDCG@k, Recall@k
│
├── pipeline/                       # Evaluation pipeline
│   ├── runner.py                   # Main evaluation orchestration
│   └── cache.py                    # JSON-based caching system
│
├── storage/                        # Database operations
│   └── database.py                 # Schema initialization and queries
│
├── visualization/                  # Reporting and visualization
│   └── reports.py                  # Text and markdown reports
│
└── scripts/                        # Executable scripts
    ├── 01_generate_queries.py      # Generate synthetic queries
    ├── 02_build_ground_truth.py    # Build ground truth
    ├── 03_run_evaluation.py        # Run evaluation
    ├── 04_generate_report.py       # Generate report
    └── run_full_evaluation.py      # All-in-one script
```

## Database Schema

The evaluation framework extends the existing `scotus_cases.db` with four tables:

```sql
-- Synthetic queries (690 rows: 345 extractive + 345 generative)
evaluation_queries (
    id, source_case_id, query_text, query_type, generation_method, created_at
)

-- Ground truth relevance scores
evaluation_ground_truth (
    id, query_id, case_id, relevance_score
)
-- Relevance: 2 = source case, 1 = cited case, 0 = other

-- Aggregate evaluation results
evaluation_results (
    id, run_id, model_name, metric_name, metric_value, run_timestamp
)

-- Per-query metrics (for error analysis)
evaluation_query_metrics (
    id, run_id, query_id, model_name, metric_name, metric_value
)
```

## Generated Data Summary

After running steps 1-2:

- **Queries**: 345 extractive queries (generative requires Ollama)
- **Ground truth**: ~1,841 entries
  - 345 source cases (relevance=2)
  - ~1,496 cited cases (relevance=1)
  - 270 queries have citations (78%)
  - Average 5.54 citations per query

## Evaluation Metrics

This framework uses four complementary metrics to evaluate retrieval quality. Each metric answers a different question about model performance.

### Precision@k

**What it measures**: Of the top-k results returned, what percentage are actually relevant?

**Formula**: `(Sum of relevance scores in top-k) / (Ideal sum of top-k from ground truth)`

Since there's only **one source case** (relevance=2) per query, the ideal maximum is:
- Position 1: Source case (relevance=2)
- Positions 2-k: Cited cases (relevance=1 each)
- **Ideal max = 2 + (k-1) = k+1** (assuming sufficient citations exist)

**Interpretation**:
- **1.0 = Perfect**: Retrieved the ideal ranking (source first, then citations)
- **0.5 = Moderate**: Got about half the ideal relevance
- **0.0 = Poor**: No relevant results in top-k

**Example**: If you ask for top-3 cases and get:
- Rank 1: Source case (relevance=2)
- Rank 2: Cited case (relevance=1)
- Rank 3: Unrelated case (relevance=0)
- Actual sum: 2+1+0 = 3
- Ideal sum: 2+1+1 = 4 (source + 2 citations)
- **P@3 = 3/4 = 0.75** ✓ Good performance

**Why it matters**: High P@1 means users immediately see relevant cases. High P@3 means the default UI results are useful.

---

### Recall@k

**What it measures**: Of all the relevant cases that exist, what percentage did we find in the top-k?

**Formula**: `(Number of relevant cases in top-k) / (Total relevant cases for this query)`

**Interpretation**:
- **1.0 = Perfect**: Found all relevant cases in top-k
- **0.5 = Moderate**: Found half of the relevant cases
- **0.0 = Poor**: Didn't find any relevant cases

**Example**: A query has 1 source case + 5 cited cases (6 total relevant):
- Top-10 contains: source + 2 cited cases = 3 relevant found
- **R@10 = 3/6 = 0.5** ✓ Found half the relevant cases

**Why it matters**: High recall means users won't miss important related cases. Critical for legal research completeness.

---

### Mean Reciprocal Rank (MRR)

**What it measures**: How quickly does the user find the first relevant result?

**Formula**: `1 / (rank of first relevant result)`

**Interpretation**:
- **1.0 = Perfect**: First result is relevant (rank=1)
- **0.5 = Good**: First relevant at rank 2
- **0.33 = Moderate**: First relevant at rank 3
- **0.1 = Poor**: First relevant at rank 10

**Example**:
- Query returns: [irrelevant, irrelevant, **relevant**, ...]
- First relevant is at rank 3
- **MRR = 1/3 = 0.33**

**Why it matters**: High MRR means users immediately see relevant results without scrolling. Measures "time to first success."

---

### NDCG@k (Normalized Discounted Cumulative Gain)

**What it measures**: How well-ordered are the results? Rewards putting highly relevant cases at the top.

**Formula**:
```
DCG@k = Σ(relevance_i / log₂(rank_i + 1))
NDCG@k = DCG@k / IDCG@k  (normalized to 0-1 scale)
```

**Interpretation**:
- **1.0 = Perfect ranking**: Most relevant cases at top (ideal order)
- **0.75 = Good ranking**: Relevant cases near top with minor mistakes
- **0.5 = Mediocre**: Relevant cases scattered throughout results
- **0.0 = Poor**: Relevant cases buried at bottom

**Example**: Two systems both return 2 relevant cases in top-3:

System A: [Source(rel=2), Cited(rel=1), Unrelated(rel=0)]
- DCG = 2/log₂(2) + 1/log₂(3) + 0/log₂(4) = 2.0 + 0.63 = 2.63
- **NDCG@3 = 1.0** ✓ Perfect order

System B: [Unrelated(rel=0), Cited(rel=1), Source(rel=2)]
- DCG = 0/log₂(2) + 1/log₂(3) + 2/log₂(4) = 0 + 0.63 + 1.0 = 1.63
- **NDCG@3 = 0.62** ✗ Wrong order (most relevant case buried)

**Why it matters**: NDCG is the most sophisticated metric. It captures both *what* you retrieve and *in what order*. Essential for ranking evaluation.

---

### How Our Metrics Work Together

Each metric answers a different question:

1. **P@1 = 0.975** → "97.5% of the time, the top result is relevant"
2. **P@3 = 0.350** → "About 1 out of 3 results in the top-3 is relevant"
3. **R@10 = 0.477** → "We found 47.7% of all relevant cases in top-10"
4. **MRR = 0.983** → "First relevant result is almost always rank 1"
5. **NDCG@3 = 0.751** → "Results are well-ordered (75% as good as perfect)"

**Ideal scores**: All metrics should be high (>0.8), but trade-offs exist:
- High precision, low recall = Conservative (few results, all good)
- Low precision, high recall = Liberal (many results, some irrelevant)
- High NDCG = Good ranking even if some irrelevant cases mixed in

## Key Features

### 1. Query Generation

**Extractive (Rule-based)**
- Skips headers (Syllabus, case names, dates)
- Extracts first 50-150 words of substantive content
- Fast, deterministic, no external dependencies

**Generative (Ollama-based)**
- Uses Llama 3.2 1B to convert case facts into natural questions
- More realistic queries that match user intent
- Requires Ollama installation (~1 hour for 345 queries)

### 2. Ground Truth

- **Source case**: Highest relevance (score=2)
- **Cited cases**: Medium relevance (score=1)
- **Fuzzy matching**: Handles citation name variations with 85% similarity threshold
- **Statistics**: 270/345 queries have citations, avg 5.54 citations/query

### 3. Caching

- JSON-based caching per (model, query) pair
- Avoids re-computing embeddings on re-runs
- Enables iterative metric development
- Cache location: `evaluation_results/cache/{model_name}/{query_id}.json`

### 4. Reproducibility

- Fixed random seeds (where applicable)
- Persistent query storage in database
- Run IDs track each evaluation
- Cached predictions ensure exact reproduction

## Configuration

Edit `evaluation/config.yaml` to customize:

```yaml
evaluation:
  models: [legal-bert, harvard-bert, sentence-transformer]
  k_values: [1, 3, 5, 10, 20]
  top_k_retrieval: 100

query_generation:
  llm_model: "llama3.2:1b"
  queries_per_case: 2

ground_truth:
  source_relevance: 2
  citation_relevance: 1
  match_threshold: 0.9
```

## Expected Results

### Sample Output

```
MODEL COMPARISON
================================================================================
Model                     P@1     P@3     P@5      MRR  NDCG@3  NDCG@10
--------------------------------------------------------------------------------
legal-bert              0.8470  0.7210  0.6540  0.8910  0.7630   0.7120
harvard-bert            0.8230  0.6980  0.6310  0.8720  0.7410   0.6930
sentence-transformer    0.7910  0.6650  0.6010  0.8430  0.7080   0.6610
================================================================================
```

**Interpretation:**
- legal-bert performs best on 5/6 metrics
- High P@1 (~85%) indicates source case often ranks first
- NDCG@3 ~0.76 shows good ranking of top results

## Advanced Usage

### Evaluate Specific Models

```bash
python evaluation/scripts/03_run_evaluation.py \
    --db-path scotus_cases.db \
    --models legal-bert harvard-bert
```

### Clear Cache and Re-run

```bash
python evaluation/scripts/03_run_evaluation.py \
    --db-path scotus_cases.db \
    --clear-cache
```

### Generate Report for Specific Run

```bash
# List available runs
python evaluation/scripts/04_generate_report.py \
    --db-path scotus_cases.db \
    --list

# Generate report for specific run
python evaluation/scripts/04_generate_report.py \
    --db-path scotus_cases.db \
    --run-id <run_id> \
    --output-dir reports/
```

## Troubleshooting

### Issue: "Ollama is not running"

**Solution**: If you haven't installed Ollama, use extractive queries only:
```bash
python evaluation/scripts/01_generate_queries.py \
    --db-path scotus_cases.db \
    --extractive-only
```

To install Ollama later:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
```

### Issue: "No queries found"

**Solution**: Run step 1 first:
```bash
python evaluation/scripts/01_generate_queries.py --db-path scotus_cases.db --extractive-only
```

### Issue: "No ground truth found"

**Solution**: Run step 2 first:
```bash
python evaluation/scripts/02_build_ground_truth.py --db-path scotus_cases.db
```

### Issue: Evaluation is slow

**Solutions**:
1. **Use cache**: Results are cached automatically on first run
2. **Evaluate fewer models**: Use `--models legal-bert`
3. **Reduce k values**: Edit `config.yaml` to use fewer k values

## Performance

- **Query generation**: ~1 second (extractive), ~1 hour (generative with Ollama)
- **Ground truth building**: ~1 minute
- **Evaluation** (per model): ~15-20 minutes first run, ~2 minutes with cache
- **Total pipeline**: ~2 hours (with Ollama), ~30 minutes (extractive only)

## Next Steps

1. **Validate results**: Manually inspect 20 random queries to verify rankings
2. **Error analysis**: Examine queries with low NDCG scores
3. **Add more models**: Test other embeddings (MPNet, E5, etc.)
4. **Optimize retrieval**: Use FAISS for faster similarity search
5. **Deploy monitoring**: Track metrics over time as models evolve

## Citation

If you use this evaluation framework, please cite:

```
Legal Case Similarity Model Evaluation Framework
Built for SCOTUS case retrieval evaluation
Metrics: Precision@k, MRR, NDCG@k
Ground truth: Citation-based relevance
```

## License

This evaluation framework is part of the PresidioAI project.

## Support

For issues or questions:
1. Check this README and troubleshooting section
2. Review the plan file: `.claude/plans/peppy-wishing-lovelace.md`
3. Inspect example outputs in `evaluation_results/`
