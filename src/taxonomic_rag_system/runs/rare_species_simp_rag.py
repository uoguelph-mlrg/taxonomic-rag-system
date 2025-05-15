"""
Module provides functionality for running a Naive Vision-Language Model (VLM).

It builds the `ImageRAGModel` class using a pre-trained VLM model from OpenAI.
on a rare species dataset and extracting taxonomic metrics and predictions.

The script performs the following steps:
1. Reads API keys from `.openai.key` and `.openrouter.key` files in the user's home dir.
2. Parses command-line arguments for output paths and a flag whether to write results.
3. Builds an ImageRAGModel and runs it on a rare species dataset.
5. Optionally writes the results (metrics and predictions) to timestamped CSV files.

Modules:
    argparse: For parsing command-line arguments.
    asyncio: For asynchronous execution of the main function.
    datetime: For generating timestamps for output files.
    os: For setting environment variables.
    pathlib.Path: For handling file paths.
    taxonomic_rag_system.core.image_rag.ImageRAGModel:
            For building and running VLM model.
    taxonomic_rag_system.utils.helpers:
            For extracting metrics and writing results to CSV.

Functions:
    _parse_arguments: Parses command-line arguments for output path and write flag.
    main: Asynchronous function that executes the rare species prediction workflow.

Usage:
    Run the script from the command line with arguments:
        python rare_species_simp_rag.py --output <path1> --vstore <path2> --write
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
    1. Sets API key env variables using `.openai.key` and `.openrouter.key` files
       located in the user's home directory.
    2. Parses command-line arguments to output paths and whether to write results.
    3. Builds a naive VLM model and runs it on a rare species dataset.
    4. Extracts taxonomic metrics and predictions from the model's output.
    5. Optionally writes results (metrics and predictions) to CSV files with timestamps.

    Args:
        None

    Returns
    -------
        None
    """
    # Set API key env variables w/ `.openai.key` and `.openrouter.key` files in home dir
    with open(Path.home() / ".openai.key", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()
    with open(Path.home() / ".openrouter.key", "r") as f:
        os.environ["OPENROUTER_API_KEY"] = f.read().strip()

    args = _parse_arguments()
    vstore_path = args.vstore
    write = args.write
    output_path = args.output

    # 1. Build Model, 2. RareSpecies Run, 3. Extract Results
    model = ImageRAGModel(vstore_path=vstore_path, model="gpt-4o")
    print(f"Device: {model.get_device()}")
    rarespp_predictions = await model.rarespecies_dataset_run(verbose=2)
    overalls, preds = extract_tax_metrics_rs(rarespp_predictions, verbose=True)

    if write:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Grab current date
        current_time = datetime.datetime.now().strftime("%H-%M-%S")  # Grab current time
        final_metrics_csv_name = str(
            output_path + f"RS_simpRAG_tax_metrics_{current_date}_{current_time}.csv"
        )
        predictions_csv_name = str(
            output_path + f"RS_simpRAG_predictions_{current_date}_{current_time}.csv"
        )
        write_preds_to_csv(preds, predictions_csv_name)
        write_overall_metrics(final_metrics_csv_name, overalls)


if __name__ == "__main__":
    asyncio.run(main())
