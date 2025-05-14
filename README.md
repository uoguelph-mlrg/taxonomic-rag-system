# Taxonomic RAG System

----------------------------------------------------------------------------------------

[![code checks](https://github.com/uoguelph-mlrg/taxonomic-rag-system/actions/workflows/code_checks.yml/badge.svg)](https://github.com/uoguelph-mlrg/taxonomic-rag-system/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/uoguelph-mlrg/taxonomic-rag-system/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/uoguelph-mlrg/taxonomic-rag-system/actions/workflows/integration_tests.yml)
[![docs](https://github.com/uoguelph-mlrg/taxonomic-rag-system/actions/workflows/docs.yml/badge.svg)](https://github.com/uoguelph-mlrg/taxonomic-rag-system/actions/workflows/docs.yml)
![GitHub License](https://img.shields.io/github/license/uoguelph-mlrg/taxonomic-rag-system)

A system for taxonomic classification and biodiversity analysis using Retrieval-Augmented Generation (RAG) and Vision-Language Models (VLMs). This repository contains the implementation of experiments presented in [Taxonomic Reasoning of Rare Arthropods: Combinging Dense Image Captioning with RAG for Interpretable Classification](https://arxiv.org/abs/2503.10886), part of the Canadian AI 2025 Conference proceedings.

## 🧑🏿‍💻 Developing

### Installing dependencies

The development environment can be set up using
[uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation). Hence, make sure it is
installed and then run:

```bash
uv sync
source .venv/bin/activate
```

In order to install dependencies for testing (codestyle, unit tests, integration tests),
run:

```bash
uv sync --dev
source .venv/bin/activate
```

### Prerequisites

Replicating experiments require an OpenAI API key, Cohere API key, and OpenRouter API key for the vision and language models. The scripts expect files in your home directory (called `.openai.key`, `.cohere.key`, `.openrouter.key`, respectively) that contain your keys. 

You can set up those files on the command line with your own API keys using the following:

Note: Below are EXAMPLE API keys and should be replaced with your actual keys.

```bash
# Ex OPENAI's apikey = "sk-1234567890abcdefg"
cat "sk-1234567890abcdefg" > $HOME/.openai.key

# Ex Cohere's apikey = "cohere-1234567890abcdefg"
cat "cohere-1234567890abcdefg" > $HOME/.cohere.key

# Ex OpenRouter's apikey = "openrouter-1234567890abcdefg"
cat "openrouter-1234567890abcdefg" > $HOME/.openrouter.key
```

##

### Running Experiments

This repository contains experiments for taxonomic classification and biodiversity analysis. 

The experiments are implemented in the `runs` module but rely on the `utils`, `core` and `preprocess` modules.

#### Preprocessing

To preprocess datasets for experiments, use the preprocessing utilities in the `preprocess` module.
Assuming the docuuments used to build the vectorstore are in `data/documents/` and you'd like to build the vectorstore to persist in `data/vstore`, you can run the following command to chunk, contextualize and vectorize the documents: 

```bash
python src/taxonomic_rag_system/preprocess/chunk_vectorize.py --source "data/documents" --pers_dir "data/vstore" --contextualize --write
```

This script will chunk, contextualize and filter the doc sources you provide (in the paper, >250K docs from Wikipedia and Wikispecies pages for Animalia) and then build a ChromaDB vector database from the informative chunks (in the paper, >550K informative chunks).

#### Utilities

The `utils` module provides helper functions and classes for tasks such as:
- Document retrieval and formatting (`retriever.py`)
- Multimodal caption and tax classification generation (`vision_models.py`)
- Dataset loading, evaluation and reporting (`helpers.py`, `evaluator.py`)
- Image processsing (`image_processing.py`)

## 

#### Rare Species Dataset Experiments

To run experiments on the rare species dataset, use the following scripts:

1. **Simple RAG Model**:
   ```bash
   python src/taxonomic_rag_system/runs/rare_species_simp_rag.py
   ```
   
2. **Advanced RAG Model**:
   ```bash
   python src/taxonomic_rag_system/runs/rare_species_adv_rag.py
   ```

3. **Naive Vision-Language Model - GPT-4o**:
   ```bash
   python src/taxonomic_rag_system/runs/rare_species_naive_gpt.py
   ```

4. **Naive Vision-Language Model - Gemini 2.0 Flash**:
   ```bash
   python src/taxonomic_rag_system/runs/rare_species_naive_gemini.py
   ```

##

### Results

The results of the experiments, including taxonomic classifications and evaluation metrics, are saved as CSV files in the specified output directory:
- Itemized classification predictions: `RS_<model>_predictions_<date>_<time>.csv`
- Rank-wise classification metrics: `RS_<model>_tax_metrics_<date>_<time>.csv`