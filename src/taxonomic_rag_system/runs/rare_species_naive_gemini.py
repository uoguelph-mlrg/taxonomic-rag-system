"""
Module providing a Naive Vision-Language Model (VLM) for taxonomic classification.

Builds the `NaiveVLModel` class using a pre-trained VLM model from OpenRouter.

It includes functionality for querying the model with images
and evaluating its performance on a rare species dataset.

Classes:
--------
- NaiveVLModel: A class for performing taxonomic classification using a VLM.

Functions:
- main(): Executes the Naive VLM run, extracts metrics, and optionally writes results.

Usage:
------
Run this script to evaluate the Naive VLM on the rare species dataset and write results:
   ```python
   rare_species_naive_gemini.py - -output_path < path > --write
   ```
"""

import argparse
import asyncio
import datetime
import os
from pathlib import Path

from taxonomic_rag_system.core.image_rag import NaiveVLModel
from taxonomic_rag_system.utils.helpers import (
    extract_tax_metrics,
    write_overall_metrics,
    write_preds_to_csv,
)


def _parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including output path, and flag for writing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        type=bool,
        help="Flag to indicate whether to write the contextualized documents.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Path where contextualized documents will be saved.",
    )

    return parser.parse_args()


async def main():
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
    output_path = args.output_path
    write = args.write

    # 1. Build Model, 2. RareSpecies Run, 3. Extract Results
    naive_model = NaiveVLModel(model="google/gemini-2.0-flash-001")
    rarespp_predictions = await naive_model.rarespecies_dataset_run(verbose=2)
    overalls, preds = extract_tax_metrics(rarespp_predictions, verbose=True)

    if write:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Grab current date
        current_time = datetime.datetime.now().strftime("%H-%M-%S")  # Grab current time
        final_metrics_csv_name = str(
            output_path
            + f"RS_naiveVLM_gemini_tax_metrics_{current_date}_{current_time}.csv"
        )
        predictions_csv_name = str(
            output_path
            + f"RS_naiveVLM_gemini_predictions_{current_date}_{current_time}.csv"
        )
        write_preds_to_csv(preds, predictions_csv_name)
        write_overall_metrics(final_metrics_csv_name, overalls)


if __name__ == "__main__":
    asyncio.run(main())
