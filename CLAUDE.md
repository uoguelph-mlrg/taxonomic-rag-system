# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a taxonomic classification and biodiversity analysis system using Retrieval-Augmented Generation (RAG) and Vision-Language Models (VLMs). The system combines image processing, descriptive captioning, and RAG-based retrieval to perform taxonomic classification of rare arthropods and other species.

This repository contains the implementation of experiments presented in [Taxonomic Reasoning of Rare Arthropods: Combining Dense Image Captioning with RAG for Interpretable Classification](https://arxiv.org/html/2503.10886v1), part of the Canadian AI 2025 Conference proceedings.

## Development Setup

### Environment Setup
```bash
# Install dependencies (requires uv to be installed)
uv sync --dev
source .venv/bin/activate

# For production dependencies only
uv sync
source .venv/bin/activate
```

### API Keys Setup
The system requires API keys stored in your home directory:
```bash
# Required API key files in home directory
~/.openai.key      # OpenAI API key
~/.cohere.key      # Cohere API key for reranking
~/.openrouter.key  # OpenRouter API key for alternative models
```

**⚠️ Important**: API keys should never be checked into version control. These files should remain in your local home directory only.

### Running Tests
```bash
# Unit tests (excludes integration tests)
uv run pytest -m "not integration_test" --cov src/taxonomic_rag_system --cov-report=xml tests

# Integration tests only
uv run pytest -m "integration_test" --cov src/taxonomic_rag_system --cov-report=xml tests

# All tests
uv run pytest tests

# Single test file
uv run pytest tests/taxonomic_rag_system/test_image_rag.py
```

### Code Quality Checks
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Individual tools
uv run ruff check src/ tests/          # Linting
uv run ruff format src/ tests/         # Formatting
uv run mypy src/                       # Type checking
```

## Architecture Overview

### Core Components

1. **Core Module (`src/taxonomic_rag_system/core/`)**
   - `image_rag.py`: Main ImageRAGModel and NaiveVLModel classes that orchestrate the entire pipeline

2. **Utils Module (`src/taxonomic_rag_system/utils/`)**
   - `retriever.py`: RAG chain building, document retrieval using Chroma and Stella embeddings
   - `vision_models.py`: Vision-language model utilities for captioning and classification
   - `image_processor.py`: Image processing utilities
   - `evaluator.py`: Evaluation metrics and dataset processing
   - `helpers.py`: String processing and formatting utilities
   - `out_models.py`: Pydantic models for structured outputs

3. **Preprocessing (`src/taxonomic_rag_system/preprocess/`)**
   - `chunk_vectorize.py`: Document chunking, contextualization, and vector database creation

4. **Experiments (`src/taxonomic_rag_system/runs/`)**
   - `rare_species_simp_rag.py`: Simple RAG model experiments
   - `rare_species_adv_rag.py`: Advanced RAG model experiments
   - `rare_species_naive_gpt.py`: Naive GPT-4o model experiments
   - `rare_species_naive_gemini.py`: Naive Gemini 2.0 Flash model experiments

### Data Flow Architecture

1. **Image Processing**: Images are processed and encoded for VLM input
2. **Caption Generation**: VLMs generate descriptive captions of images
3. **Document Retrieval**: RAG system retrieves relevant taxonomic documents based on captions
4. **Classification**: Final taxonomic classification using retrieved context
5. **Evaluation**: Results are evaluated against ground truth taxonomic labels

### Key Dependencies

- **[LangChain](https://langchain.com)**: RAG pipeline construction and document processing
- **[ChromaDB](https://www.trychroma.com)**: Vector database for document storage and retrieval
- **[OpenAI](https://openai.com)/[OpenRouter](https://openrouter.ai)**: Vision-language model APIs
- **[Cohere](https://cohere.com)**: Document reranking
- **[Hugging Face](https://huggingface.co)**: Stella embeddings for document vectorization
- **[PyTorch](https://pytorch.org)**: Deep learning operations
- **[Instructor](https://useinstructor.com)**: Structured output parsing from LLMs

## Common Development Tasks

### Running Experiments

**⚠️ Cost Warning**: Experiments involve API calls to OpenAI, OpenRouter, and Cohere services which incur costs. Do not launch non-trivial experiments without budget approval.

#### Preprocessing Documents
```bash
# Chunk, contextualize and vectorize documents
python src/taxonomic_rag_system/preprocess/chunk_vectorize.py \
    --source "data/documents" \
    --pers_dir "data/vstore" \
    --contextualize --write
```

#### Rare Species Classification Experiments
```bash
# Simple RAG model
python src/taxonomic_rag_system/runs/rare_species_simp_rag.py \
    --output <output_directory> --vstore <persistence_directory> --write

# Advanced RAG model
python src/taxonomic_rag_system/runs/rare_species_adv_rag.py

# Naive GPT-4o model
python src/taxonomic_rag_system/runs/rare_species_naive_gpt.py

# Naive Gemini 2.0 Flash model
python src/taxonomic_rag_system/runs/rare_species_naive_gemini.py \
    --output <output_directory> --write
```

### Testing Strategy

- **Unit Tests**: Mock external APIs and test individual components in isolation
- **Integration Tests**: Test full pipeline with real API calls (marked with `@pytest.mark.integration_test`)
- **Fixtures**: Comprehensive mock fixtures in `tests/conftest.py` for consistent testing

### Code Style and Standards

- **Python Version**: 3.12
- **Line Length**: 88 characters
- **Docstring Style**: NumPy convention
- **Type Checking**: MyPy with strict configuration
- **Linting**: Ruff with extensive rule set including flake8 plugins
- **Pre-commit Hooks**: Comprehensive checks including security scanning and typo detection

### Output Formats

Experiment results are saved as CSV files with timestamped filenames:
- `RS_<model>_predictions_<date>_<time>.csv`: Individual predictions
- `RS_<model>_tax_metrics_<date>_<time>.csv`: Taxonomic classification metrics

### API Integration Patterns

The system integrates with multiple AI service providers:
- **OpenAI**: Primary VLM and LLM provider
- **OpenRouter**: Alternative model access
- **Cohere**: Document reranking services

All API clients are configured with async support and proper error handling.