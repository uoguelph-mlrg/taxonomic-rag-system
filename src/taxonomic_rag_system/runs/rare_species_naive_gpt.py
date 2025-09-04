"""
Module providing main entry point for a Naive VLM for taxonomic classification.

Builds the `NaiveVLModel` class using a pre-trained VLM model from OpenAI's GPT-4o.

This script runs the model on imageomic's rare species dataset and
extracts taxonomic predictions as well as evaluation of performance.

Classes:
--------
NaiveVLModel: A class for taxonomic classification using GPT-4o.

Functions:
--------
_parse_arguments: Parses command-line arguments for output path and write flag.
main(): Executes the Naive VLM run, extracts metrics, and optionally writes results.

Usage:
------
Run this script to evaluate the Naive VLM on the rare species dataset and write results:
   ```python
   rare_species_naive_gpt.py - -output_path < path1 > --write
   ```
"""

import argparse
import asyncio
import datetime
import json
from pathlib import Path

from taxonomic_rag_system.core.image_rag import NaiveVLModel
from taxonomic_rag_system.utils.helpers import (
    extract_tax_metrics,
    write_overall_metrics,
    write_preds_to_csv,
    write_sample_binary_accuracy_csv,
    write_rank_attempts_csv,
    filter_nonempty_results,
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

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")

    # 1. Build Model
    naive_model = NaiveVLModel(model="gpt-4o", openrouter=False)
    # 2. RareSpecies Run
    rarespp_predictions = await naive_model.rarespecies_dataset_run(
        interval=interval, verbose=2
    )
    # 3. Extract Results
    overalls, preds = extract_tax_metrics(rarespp_predictions, verbose=True)

    if write:
        # Create output directory if it doesn't exist
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)

        final_metrics_csv_name = str(
            output_path
            + f"RS_naiveVLM_gpt_tax_metrics_{current_date}_{current_time}.csv"
        )
        predictions_csv_name = str(
            output_path
            + f"RS_naiveVLM_gpt_predictions_{current_date}_{current_time}.csv"
        )
        # Write per-sample binary accuracy labels
        sample_binary_csv_name = str(
            output_path
            + f"RS_naiveVLM_gpt_sample_binary_accuracy_{current_date}_{current_time}.csv"
        )
        # Write rank-level attempts (Count)
        rank_attempts_csv_name = str(
            output_path
            + f"RS_naiveVLM_gpt_rank_attempts_{current_date}_{current_time}.csv"
        )
        write_preds_to_csv(preds, predictions_csv_name)
        write_overall_metrics(final_metrics_csv_name, overalls)
        write_sample_binary_accuracy_csv(preds, sample_binary_csv_name)
        write_rank_attempts_csv(rank_attempts_csv_name, overalls, total_samples=len(preds))

        # Write filtered results (exclude samples with empty predictions) to outputs_uq_data_collection/
        uq_dir_path = Path(output_path).parent / "outputs_uq_data_collection"
        uq_dir_path.mkdir(parents=True, exist_ok=True)

        filtered_results = filter_nonempty_results(rarespp_predictions)
        overalls_uq, preds_uq = extract_tax_metrics(filtered_results, verbose=True)
        # Enrich preds_uq with RSID from filtered_results for downstream CSV writers
        rsids_uq = [s.get("RSID") for s in filtered_results]
        for g, r in zip(preds_uq, rsids_uq):
            g["RSID"] = r

        final_metrics_csv_name_uq = str((uq_dir_path / f"RS_naiveVLM_gpt_tax_metrics_{current_date}_{current_time}.csv").as_posix())
        predictions_csv_name_uq = str((uq_dir_path / f"RS_naiveVLM_gpt_predictions_{current_date}_{current_time}.csv").as_posix())
        sample_binary_csv_name_uq = str((uq_dir_path / f"RS_naiveVLM_gpt_sample_binary_accuracy_{current_date}_{current_time}.csv").as_posix())
        rank_attempts_csv_name_uq = str((uq_dir_path / f"RS_naiveVLM_gpt_rank_attempts_{current_date}_{current_time}.csv").as_posix())
        write_preds_to_csv(preds_uq, predictions_csv_name_uq)
        write_overall_metrics(final_metrics_csv_name_uq, overalls_uq)
        write_sample_binary_accuracy_csv(preds_uq, sample_binary_csv_name_uq)
        write_rank_attempts_csv(rank_attempts_csv_name_uq, overalls_uq, total_samples=len(preds_uq))

        # Write filtered JSONL with prompt/response pairs to UQ directory
        # For naive runs there may not be a prompt JSONL; only write if present
        try:
            # If a JSONL matching naming exists in outputs/, filter it by RSID
            # Discover any *_prompt_response_pairs_*.jsonl from outputs/ with same timestamp prefix
            # Not strictly necessary, so keep it best-effort and silent on failure
            filtered_rsids = {r for r in rsids_uq if r}
            # Example expected name (if produced upstream): RS_naiveVLM_gpt_prompt_response_pairs_...
            candidate = str(
                (Path(output_path) / f"RS_naiveVLM_gpt_prompt_response_pairs_{current_date}_{current_time}.jsonl").as_posix()
            )
            if Path(candidate).exists():
                uq_jsonl_name = str(
                    (uq_dir_path / f"RS_naiveVLM_gpt_prompt_response_pairs_{current_date}_{current_time}.jsonl").as_posix()
                )
                with open(candidate, "r", encoding="utf-8") as src, open(
                    uq_jsonl_name, "w", encoding="utf-8"
                ) as dst:
                    for line in src:
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if obj.get("RSID") in filtered_rsids:
                            dst.write(line)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
