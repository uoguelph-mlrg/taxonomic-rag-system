# Taxonomic RAG System Documentation

Welcome to the documentation for the Taxonomic RAG System, a framework for taxonomic classification and biodiversity analysis. This system integrates image processing, Retrieval-Augmented Generation (RAG), and Vision-Language Models (VLMs) to classify rare species and provide contextualized insights.

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Modules](#modules)
  - [Core](#core)
  - [Preprocess](#preprocess)
  - [Runs](#runs)
  - [Utils](#utils)
- [API Reference](api.md)
- [User Guide](user_guide.md)

---

## Overview

The Taxonomic RAG System combines RAG with Wikipedia and Wikispecies data to:
1. Classify rare species with taxonomic reasoning.
2. Generate commentary on biodiversity and classification confidence.

It leverages VLMs to generate feature-based descriptions, which are semantically matched to vectorstore documents for enhanced classification accuracy.

---

## Getting Started

Follow the [User Guide](user_guide.md) for installation and usage instructions. Ensure all dependencies are installed and configured.

---

## Modules

### **Core**
Implements the main logic for taxonomic classification, including image processing, caption generation, and RAG-based retrieval.

### **Preprocess**
Provides utilities for preparing datasets, including chunking, contextualizing, and building vectorstores.

### **Runs**
Contains scripts for running experiments, such as rare species classification using RAG and VLM models.

### **Utils**
Offers helper functions for document retrieval, evaluation metrics, image processing, and taxonomic modeling.

---

For detailed information, refer to the [API Reference](api.md) and [User Guide](user_guide.md).
