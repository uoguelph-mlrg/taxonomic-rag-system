# Upstream: `taxonomic-rag-system` (`feature/uq_draft`) — Repro & outputs

- **Branch**: `https://github.com/uoguelph-mlrg/taxonomic-rag-system/tree/feature/uq_draft`
- **Goal**: help users reproduce runs on this branch (Rare Species dataset, Arthropoda subset) and understand **exactly what files are written** and what they contain.

## Prerequisites (minimal)

- **Python env** (recommended: `uv`):

```bash
git clone https://github.com/uoguelph-mlrg/taxonomic-rag-system.git
cd taxonomic-rag-system
git checkout feature/uq_draft

uv sync --dev
source .venv/bin/activate
```

- **API keys (files in your home dir)** (loaded by `src/taxonomic_rag_system/utils/helpers.py:load_api_keys()`):
  - **Required**: `~/.openai.key` (Simple/Advanced RAG, Naive GPT-4o)
  - **Optional**: `~/.openrouter.key` (needed for Naive Gemini / OpenRouter models)
  - **Optional**: `~/.cohere.key` (needed for Advanced RAG reranking)

- **Dataset**: auto-downloaded and cached via HuggingFace `datasets.load_dataset("imageomics/rare-species")` (first run needs network).
  - The evaluator **filters to `phylum == "Arthropoda"` only** (see `src/taxonomic_rag_system/utils/evaluator.py`).

- **Vector store (RAG only)**: `--vstore` points to a Chroma persistence directory (often contains `chroma.sqlite3`).
  - Typical build command (from `README.md`):

```bash
python src/taxonomic_rag_system/preprocess/chunk_vectorize.py \
  --source "data/documents" \
  --pers_dir "data/vstore" \
  --contextualize --write
```

## Quickstart (small interval + write outputs)

> Important: several entry scripts build filenames by string concatenation; **make sure your output directory ends with `/`**, e.g. `./outputs/`, otherwise you may get paths like `./outputsRS_...`.

```bash
python -u src/taxonomic_rag_system/runs/rare_species_simp_rag.py \
  --vstore "/path/to/chroma" \
  --output "./outputs/" \
  --write \
  --interval-start 0 \
  --interval-end 10
```

## Interval semantics (critical)

Intervals are applied in `src/taxonomic_rag_system/utils/evaluator.py` as:

- `select(range(start, end))`, so **`interval-end` is exclusive**.
  - Example: `--interval-start 0 --interval-end 59` selects **0..58**.

Also, because of Arthropoda filtering, the final sample count may be **< (end - start)**.

## Entry points (how to run)

### Simple RAG (gpt-4o + vectorstore)

```bash
python -u src/taxonomic_rag_system/runs/rare_species_simp_rag.py \
  --vstore "/path/to/chroma" \
  --output "./outputs/" \
  --write \
  --interval-start 0 --interval-end 100
```

Optional (multi-sampling; only generates extra multisample files when enabled):

```bash
python -u src/taxonomic_rag_system/runs/rare_species_simp_rag.py \
  --vstore "/path/to/chroma" \
  --output "./outputs/" \
  --write \
  --multi-sampling --num-samples 5 \
  --interval-start 0 --interval-end 100
```

### Advanced RAG (mmr + rerank + multiquery)

```bash
python -u src/taxonomic_rag_system/runs/rare_species_adv_rag.py \
  --vstore "/path/to/chroma" \
  --output "./outputs/" \
  --write \
  --interval-start 0 --interval-end 100
```

### Naive GPT-4o (no retrieval; VLM-only classification)

```bash
python -u src/taxonomic_rag_system/runs/rare_species_naive_gpt.py \
  --output_path "./outputs/" \
  --write \
  --interval-start 0 --interval-end 100
```

### Naive Gemini (OpenRouter; no retrieval)

```bash
python -u src/taxonomic_rag_system/runs/rare_species_naive_gemini.py \
  --output_path "./outputs/" \
  --write \
  --interval-start 0 --interval-end 100
```

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
  - Simple/Advanced RAG: lines contain `rsid/prompt/response` (the script rewrites the JSONL to a minimal schema after writing).
  - Naive GPT/Gemini: also writes a prompt/response-style JSONL (produced inside the script).
- `RS_simpRAG_logprobs_*.jsonl`
  - Simple RAG writes token logprobs when `--write` is enabled.
- `RS_advRAG_logprobs_*.jsonl`
  - Advanced RAG writes token logprobs when `--write` is enabled.
- `RS_naiveVLM_gpt_logprobs_*.jsonl`
  - Naive GPT-4o writes token logprobs when `--write` is enabled.
  - Naive Gemini does not currently write logprobs JSONL.

### D) Exact filename patterns per entry script

#### Simple RAG (`rare_species_simp_rag.py`, with `--write`)

- `RS_simpRAG_predictions_<date>_<time>.csv`
- `RS_simpRAG_tax_metrics_hierarchical_<date>_<time>.csv`
- `RS_simpRAG_rank_attempts_<date>_<time>.csv`
- `RS_simpRAG_per_rank_binary_<date>_<time>.jsonl`
- `RS_simpRAG_prompt_response_pairs_<date>_<time>.jsonl`
- `RS_simpRAG_logprobs_<date>_<time>.jsonl`

Multi-sampling extras (only with `--multi-sampling`):

- `RS_simpRAG_multisample_prompts_n{num_samples}_<date>_<time>.jsonl`
- `RS_simpRAG_multisample_base_response_<date>_<time>.jsonl`
- `RS_simpRAG_multisample_samples_n{num_samples}_<date>_<time>.jsonl`

#### Advanced RAG (`rare_species_adv_rag.py`, with `--write`)

- `RS_advRAG_predictions_<date>_<time>.csv`
- `RS_advRAG_tax_metrics_hierarchical_<date>_<time>.csv`
- `RS_advRAG_rank_attempts_<date>_<time>.csv`
- `RS_advRAG_per_rank_binary_<date>_<time>.jsonl`
- `RS_advRAG_prompt_response_pairs_<date>_<time>.jsonl`
- `RS_advRAG_logprobs_<date>_<time>.jsonl`

#### Naive GPT-4o (`rare_species_naive_gpt.py`, with `--write`)

- `RS_naiveVLM_gpt_predictions_<date>_<time>.csv`
- `RS_naiveVLM_gpt_tax_metrics_hierarchical_<date>_<time>.csv`
- `RS_naiveVLM_gpt_rank_attempts_<date>_<time>.csv`
- `RS_naiveVLM_gpt_per_rank_binary_<date>_<time>.jsonl`
- `RS_naiveVLM_gpt_prompt_response_pairs_<date>_<time>.jsonl`
- `RS_naiveVLM_gpt_logprobs_<date>_<time>.jsonl`

#### Naive Gemini (`rare_species_naive_gemini.py`, with `--write`)

- `RS_naiveVLM_gemini_predictions_<date>_<time>.csv`
- `RS_naiveVLM_gemini_tax_metrics_hierarchical_<date>_<time>.csv`
- `RS_naiveVLM_gemini_rank_attempts_<date>_<time>.csv`
- `RS_naiveVLM_gemini_per_rank_binary_<date>_<time>.jsonl`
- `RS_naiveVLM_gemini_prompt_response_pairs_<date>_<time>.jsonl`

## (Optional) Extra outputs from Slurm/HPC submit scripts

If you run via a wrapper like `submit_jobs_rag_model_uq_collect_data_killarney_l40s.sh`, you typically also get:

- **SLURM stdout/stderr** logs, e.g. `slogs_.../%A.out`, `slogs_.../%A.err`
- **GPU monitoring CSV** (if the wrapper runs `nvidia-smi -l ... -f ...`), e.g. `slogs_.../<SLURM_JOB_ID>-gpu.csv`

