# API Reference

## Utils Module

::: taxonomic_rag_system.utils
    options:
      show_root_heading: true
      show_root_full_path: true

The `utils` module provides helper functions and classes for various tasks, including:

**Document Retrieval, LLM Chains and Contextualization**:
  - `retriever.py`: Build models with chains for retrieval and passing context to LLMs.

**Vision Models**:
  - `vision_models.py`: Generate captions, classifications and commentary for images using Vision-Language Models (VLMs).

**Dataset Loading, Evaluation, and Reporting**:
  - `helpers.py`: Compute metrics (e.g., accuracy, F1 scores), generate classification reports, and export results to CSV.
  - `evaluator.py`: Load and build dataloaders from rare species dataset and evaluate classification performance.

  **Image Processing**:
  - `image_processing.py`: Handle image conversions (e.g., Base64, PIL).

## Preprocess Module

::: taxonomic_rag_system.preprocess
    options:
      show_root_heading: true
      show_root_full_path: true

The `preprocess` module provides utilities for preparing datasets for experiments. Key functionalities include:

- Chunking, contextualizing, and filtering document sources.
- Building a ChromaDB vector database from informative chunks.

Example usage:
```bash
python src/taxonomic_rag_system/preprocess/chunk_vectorize.py --source <source_directory> --output_path <output_directory> --pers_dir <persistence_directory> --contextualize --write
```

## Runs Module

::: taxonomic_rag_system.runs
    options:
      show_root_heading: true
      show_root_full_path: true

The `runs` module provides scripts for executing experiments from the paper including:

**Naive VLM**:
  - `rare_species_naive_gemini.py`: Inference with Gemini 2.0 Flash with OpenRouter for taxonomic reasoning and classifier of images in rare species dataset.

**RAG Model**:
  - `rare_species_simp_rag.py`: Inference with Simple RAG Model using OpenAI for taxonomic reasoning and classifier of images in rare species dataset, using Wikipedia and Wikispecies documents.

Example usage:
```bash
python src/taxonomic_rag_system/runs/rare_species_simp_rag.py --output <output_directory> --vstore <persistence_directory> --write
```
