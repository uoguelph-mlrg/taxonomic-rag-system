"""
Module providing main entry point for a Naive VLM for taxonomic classification.

Builds the `NaiveVLModel` class using a pre-trained VLM model from Google's Deepmind
using OpenRouter.

This script runs the model on imageomic's rare species dataset and
extracts taxonomic predictions as well as evaluation of performance.

Classes:
--------
NaiveVLModel: A class for taxonomic classification using Gemini 2.0 Flash.

Functions:
--------
_parse_arguments: Parses command-line arguments for output path and write flag.
main(): Executes the Naive VLM run, extracts metrics, and optionally writes results.

Usage:
------
Run this script to evaluate the Naive VLM on the rare species dataset and write results:
   ```python
   rare_species_naive_gemini.py - -output_path < path1 > --write
   ```
"""

import argparse
import asyncio
import datetime
from pathlib import Path

from taxonomic_rag_system.core.image_rag import NaiveVLModel
from taxonomic_rag_system.utils.helpers import (
    extract_tax_metrics,
    write_overall_metrics,
    write_preds_to_csv,
)


def _parse_arguments() -> argparse.Namespace:
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
        help="Flag to indicate whether to write the contextualized documents.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Path where contextualized documents will be saved.",
    )
    parser.add_argument(
        "--interval-start",
        type=int,
        default=0,
        help="Start index in rare-species dataset (default: 0)",
    )
    parser.add_argument(
        "--interval-end",
        type=int,
        default=999,
        help="End index in rare-species dataset (default: 999)",
    )

    return parser.parse_args()


async def main() -> None:
    """
    Execute the Naive VLM run for rare species predictions with output metrics.

    Performs the following steps:
    1. Parses command-line arguments to output paths and whether to write results.
    2. Builds a naive VLM model and runs it on a rare species dataset.
    3. Extracts taxonomic metrics and predictions from the model's output.
    4. Optionally writes results (metrics and predictions) to CSV files with timestamps.

    Args:
        None

    Returns
    -------
        None
    """
    args = _parse_arguments()
    output_path = args.output_path
    write = args.write
    interval = (args.interval_start, args.interval_end)

    # 1. Build Model, 2. RareSpecies Run, 3. Extract Results
    naive_model = NaiveVLModel(model="google/gemini-2.0-flash-001")
    rarespp_predictions = await naive_model.rarespecies_dataset_run(
        interval=interval, verbose=2
    )
    rank_metrics, overall_metrics, preds = extract_tax_metrics(
        rarespp_predictions, verbose=True
    )

    if write:
        # Create output directory if it doesn't exist
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)

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
        write_overall_metrics(final_metrics_csv_name, rank_metrics, overall_metrics)


if __name__ == "__main__":
    asyncio.run(main())
