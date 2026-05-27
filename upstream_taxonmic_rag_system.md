# Upstream: `taxonomic-rag-system` (`feature/uq_draft`) 

- **Branch**: `https://github.com/uoguelph-mlrg/taxonomic-rag-system/tree/feature/uq_draft`
- **Supported pipelines on this branch**:
  - **Simple RAG**: `src/taxonomic_rag_system/runs/rare_species_simp_rag.py`
  - **Naive GPT-4o**: `src/taxonomic_rag_system/runs/rare_species_naive_gpt.py`

## Prerequisites (minimal)

- **Python env** (recommended: `uv`):

```bash
git clone https://github.com/uoguelph-mlrg/taxonomic-rag-system.git
cd taxonomic-rag-system
git checkout feature/uq_draft

uv sync --extra gpu
source .venv/bin/activate
```

- **API keys (files in your home dir)** (loaded by `src/taxonomic_rag_system/utils/helpers.py:load_api_keys()`):
  - **Required**: `~/.openai.key`

- **Dataset**: auto-downloaded and cached via HuggingFace `datasets.load_dataset("imageomics/rare-species")` (first run needs network).
  - The evaluator **filters to `phylum == "Arthropoda"` only** (see `src/taxonomic_rag_system/utils/evaluator.py`).

## Quickstart (small interval + write outputs)

> Important: several entry scripts build filenames by string concatenation; **make sure your output directory ends with `/`**, e.g. `./outputs/`, otherwise you may get paths like `./outputsRS_...`.


For Simple RAG:
```bash
python src/taxonomic_rag_system/runs/rare_species_simp_rag.py \
  --vstore "/path/to/chroma" \
  --output "./outputs/" \
  --write \
  --interval-start 0 \
  --interval-end 10
```

For Naive GPT:
```bash
python src/taxonomic_rag_system/runs/rare_species_naive_gpt.py \
  --output "./outputs/" \
  --write \
  --interval-start 0 \
  --interval-end 10
```

## Interval semantics (critical)

Intervals are applied in `src/taxonomic_rag_system/utils/evaluator.py` as:

- `select(range(start, end))`, so **`interval-end` is exclusive**.
  - Example: `--interval-start 0 --interval-end 59` selects **0..58**.

Note: 11982 is the last index of the rare-species dataset, so `--interval-end 11983` is the full dataset.

## Outputs (where they go, what they are)

Assume your output directory is `OUTPUT_DIR` (e.g. `./outputs/`). Each run uses a timestamp suffix: `YYYY-MM-DD_HH-MM-SS`.

### A) Common CSVs (predictions + metrics)

All `*_predictions_*.csv` files are written by `src/taxonomic_rag_system/utils/helpers.py:write_preds_to_csv()` with columns:

- `RSID, Kingdom, Phylum, Class, Order, Family, Genus, Species`
- plus (if present) `BinaryAccuracy`: computed on *predicted-only* ranks (see `helpers.py:sample_binary_accuracy()`).

All `*_tax_metrics_hierarchical_*.csv` are written by `helpers.py:write_overall_metrics()`:

- one row per rank with `Accuracy, F1`
- plus hierarchical columns `HP, HR, HF` (Hierarchical Precision/Recall/F1).

`*_rank_attempts_*.csv` are written by `helpers.py:write_rank_attempts_csv()`:

- one row per rank showing attempted count and percent (`Attempts (%)`).

### B) Per-rank JSONL (strict binary labels + per-rank hierarchical accuracy)

`*_per_rank_binary_*.jsonl` is written by `helpers.py:write_per_rank_binary_jsonl()` with **one line per (sample, rank)**. Key fields:

- `rsid`: sample ID
- `rank_idx`: 1..7 (Kingdom..Species)
- `rank`: rank name
- `pred` / `gold`: predicted / gold strings (missing predictions may be `N/A`)
- `attempted`: whether this rank was attempted (prediction exists and is not `N/A`)
- `ancestor_ok_upto_parent`: strict prefix correctness up to the parent rank
- `binary_accuracy_current_rank`: strict 0/1 (attempted AND pred==gold AND ancestors correct)
- `hierarchical_accuracy_rank`: HF at this rank (blank if not attempted)

### C) Prompt/Response logs (+ optional logprobs)

- `*_prompt_response_pairs_*.jsonl`
  - Lines contain `rsid/prompt/response` (plus lightweight metadata) for run traceability.
- `RS_simpRAG_logprobs_*.jsonl`
  - Simple RAG token-level logprobs JSONL for the final classification call (UQ).
- `RS_naiveVLM_gpt_logprobs_*.jsonl`
  - Naive GPT-4o token-level logprobs JSONL for the classification call (UQ).

### D) Exact filename patterns per entry script

#### Simple RAG (`rare_species_simp_rag.py`, with `--write`)

- `RS_simpRAG_predictions_<date>_<time>.csv`: per-sample taxonomic predictions (RSID + ranks; includes `BinaryAccuracy` when available).
- `RS_simpRAG_tax_metrics_hierarchical_<date>_<time>.csv`: per-rank Accuracy/F1 plus hierarchical HP/HR/HF summary.
- `RS_simpRAG_rank_attempts_<date>_<time>.csv`: how often each rank was attempted (non-`N/A`) across the run.
- `RS_simpRAG_per_rank_binary_<date>_<time>.jsonl`: one line per (sample, rank) with strict binary correctness and hierarchical accuracy.
- `RS_simpRAG_prompt_response_pairs_<date>_<time>.jsonl`: prompt/response trace for each sample (for auditing/replay).
- `RS_simpRAG_logprobs_<date>_<time>.jsonl`: token-level logprobs for the final classification call (for UQ).


#### Naive GPT-4o (`rare_species_naive_gpt.py`, with `--write`)

- `RS_naiveVLM_gpt_predictions_<date>_<time>.csv`: per-sample taxonomic predictions from the naive VLM (includes `BinaryAccuracy`).
- `RS_naiveVLM_gpt_tax_metrics_hierarchical_<date>_<time>.csv`: per-rank Accuracy/F1 plus hierarchical HP/HR/HF summary.
- `RS_naiveVLM_gpt_rank_attempts_<date>_<time>.csv`: how often each rank was attempted (non-`N/A`) across the run.
- `RS_naiveVLM_gpt_per_rank_binary_<date>_<time>.jsonl`: one line per (sample, rank) with strict binary correctness and hierarchical accuracy.
- `RS_naiveVLM_gpt_prompt_response_pairs_<date>_<time>.jsonl`: prompt/response trace for each sample (for auditing/replay).
- `RS_naiveVLM_gpt_logprobs_<date>_<time>.jsonl`: token-level logprobs for the classification call (for UQ).