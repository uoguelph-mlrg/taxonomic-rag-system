# User Guide

## Overview

The Taxonomic RAG System combines image captioning and Retrieval-Augmented Generation (RAG) using rich text-based biodiversity knowledge bases to classify and analyze biodiversity data of rare arthropod images. This system is designed to facilitate taxonomic reasoning and enhance the interpretability of classification results.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/uoguelph-mlrg/taxonomic-rag-system.git
   cd taxonomic-rag-system
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

3. Install additional dependency groups if needed:
   ```bash
   uv sync --group docs --group dev
   ```

## Running Scripts

After completing the installation, you can run the provided scripts for experiments and analysis. Below are the instructions for running key scripts:

### Preprocessing

To prepare a vectorstore from documents, use the `chunk_vectorize.py` script:
```bash
python src/taxonomic_rag_system/preprocess/chunk_vectorize.py --source <source_directory> --pers_dir <persistence_directory> --contextualize --write
```
- `--source`: Path to the directory containing document sources.
- `--pers_dir`: Path to the persistent directory for the vector store.
- `--contextualize`: Flag to enable contextualization of documents (using OpenAI's gpt-4o-mini).
- `--write`: Flag to save the processed data.

### Rare Species Dataset Experiments

1. **Simple RAG Model**:
   ```bash
   python src/taxonomic_rag_system/runs/rare_species_simp_rag.py --output <output_directory> --vstore <persistence_directory> --write
   ```
   - `--output`: Path to save the results.
   - `--vstore`: Path to the persistent directory for the vector store.
   - `--write`: Flag to save the results.

2. **Naive Vision-Language Model - Gemini 2.0 Flash**:
   ```bash
   python src/taxonomic_rag_system/runs/rare_species_naive_gemini.py --output <output_directory> --write
   ```
   - `--output`: Path to save the results.
   - `--write`: Flag to save the results.

### Results

The results of the experiments are saved as CSV files in the specified `output_directory`:
- Predictions: `RS_<model>_predictions_<date>_<time>.csv`
- Metrics: `RS_<model>_tax_metrics_<date>_<time>.csv`
