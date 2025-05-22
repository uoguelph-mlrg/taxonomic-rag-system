"""
Module providing main entry point for a Advanced RAG Model for taxonomic classification.

It builds the `ImageRAGModel` class using a pre-trained VLM model from OpenAI and
a vector database build from Wiki documents. It also performs reranking using Cohere's
Reranker and multiquery using another LLM pass with an OpenAI model.

This script runs the model on imageomic's rare species dataset and
extracts taxonomic predictions as well as evaluation of performance.

The script performs the following steps:
1. Parses command-line arguments for output paths and a flag whether to write results.
2. Builds an ImageRAGModel and runs it on a rare species dataset.
3. Optionally writes the results (metrics and predictions) to timestamped CSV files.

Modules:
--------
    argparse: For parsing command-line arguments.
    asyncio: For asynchronous execution of the main function.
    datetime: For generating timestamps for output files.
    taxonomic_rag_system.core.image_rag.ImageRAGModel:
            For building and running model.
    taxonomic_rag_system.utils.helpers:
            For extracting metrics and writing results to CSV.

Functions:
--------
    _parse_arguments: Parses command-line args for write, output, and vectorstore paths.
    main: Executes the Advanced RAG run, extracts metrics, and optionally writes results.

Usage:
--------
    Run the script from the command line with arguments:
        ```python
        rare_species_adv_rag.py - -output < path1 > --vstore < path2 > --write
        ```
"""

import argparse
import asyncio
import datetime
import os
from pathlib import Path

from taxonomic_rag_system.core.image_rag import ImageRAGModel

# Local imports
from taxonomic_rag_system.utils.helpers import (
    extract_tax_metrics_rs,
    write_overall_metrics,
    write_preds_to_csv,
)


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including output and vstore paths and flag for writing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vstore",
        type=str,
        help="Path to the vector store. (/chroma/ directory)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        type=bool,
        help="Flag to indicate whether to write the contextualized documents.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path where contextualized documents will be saved.",
    )

    return parser.parse_args()


async def main() -> None:
    """
    Execute the Naive VLM run for rare species predictions with output metrics.

    This function performs the following steps:
    1. Parses command-line arguments to output paths and whether to write results.
    2. Builds Advanced RAG Model and runs it on a rare species dataset.
    3. Extracts taxonomic metrics and predictions from the model's output.
    4. Optionally writes results (metrics and predictions) to CSV files with timestamps.

    Args:
        None

    Returns
    -------
        None
    """
    args = _parse_arguments()
    vstore_path = args.vstore
    write = args.write
    output_path = args.output

    with open(Path.home() / ".cohere.key", "r") as f:
        os.environ["COHERE_API_KEY"] = f.read().strip()

    # 1. Build Model
    model = ImageRAGModel(
        vstore_path=vstore_path,
        model="gpt-4o",
        search_type="mmr",
        k=10,
        rerank=True,
        multiquery=True,
    )
    print(f"Device: {model.get_device()}")
    # 2. RareSpecies Run
    rarespp_predictions = await model.rarespecies_dataset_run(verbose=2)
    # 3. Extract Results
    overalls, preds = extract_tax_metrics_rs(rarespp_predictions, verbose=True)

    if write:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Grab current date
        current_time = datetime.datetime.now().strftime("%H-%M-%S")  # Grab current time
        final_metrics_csv_name = str(
            output_path + f"RS_advRAG_tax_metrics_{current_date}_{current_time}.csv"
        )
        predictions_csv_name = str(
            output_path + f"RS_advRAG_predictions_{current_date}_{current_time}.csv"
        )
        write_preds_to_csv(preds, predictions_csv_name)
        write_overall_metrics(final_metrics_csv_name, overalls)


if __name__ == "__main__":
    asyncio.run(main())
