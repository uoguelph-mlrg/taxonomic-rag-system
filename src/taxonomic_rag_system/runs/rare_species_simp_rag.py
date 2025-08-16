"""
Module providing main entry point for a Simple RAG Model for taxonomic classification.

It builds the `ImageRAGModel` class using a pre-trained VLM model from OpenAI and
a vector database build from Wiki documents.

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
    main: Executes the Simple RAG run, extracts metrics, and optionally writes results.

Usage:
--------
    Run the script from the command line with arguments:
        ```python
        rare_species_simp_rag.py - -output < path1 > --vstore < path2 > --write
        ```
"""

import argparse
import asyncio
import datetime
import json
from pathlib import Path

from taxonomic_rag_system.core.image_rag import ImageRAGModel

# Local imports
from taxonomic_rag_system.utils.helpers import (
    extract_tax_metrics_rs,
    write_overall_metrics,
    write_preds_to_csv,
    write_sample_binary_accuracy_csv,
    write_rank_attempts_csv,
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
        help="Flag to indicate whether to write the contextualized documents.",
    )
    parser.add_argument(
        "--output",
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

    This function performs the following steps:
    1. Parses command-line arguments to output paths and whether to write results.
    2. Builds Simple RAG Model and runs it on a rare species dataset.
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
    interval = (args.interval_start, args.interval_end)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    prompt_jsonl_name = str(
        output_path + f"RS_simpRAG_prompt_response_pairs_{current_date}_{current_time}.jsonl"
    )

    # 1. Build Model, 2. RareSpecies Run, 3. Extract Results
    model = ImageRAGModel(vstore_path=vstore_path, model="gpt-4o", prompt_log_path=prompt_jsonl_name if write else None)
    print(f"Device: {model.get_device()}")
    rarespp_predictions = await model.rarespecies_dataset_run(interval=interval, verbose=2)
    overalls, preds = extract_tax_metrics_rs(rarespp_predictions, verbose=True)

    if write:
        # Create output directory if it doesn't exist
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)

        final_metrics_csv_name = str(
            output_path + f"RS_simpRAG_tax_metrics_{current_date}_{current_time}.csv"
        )
        predictions_csv_name = str(
            output_path + f"RS_simpRAG_predictions_{current_date}_{current_time}.csv"
        )
        # Write per-sample binary accuracy labels
        sample_binary_csv_name = str(
            output_path + f"RS_simpRAG_sample_binary_accuracy_{current_date}_{current_time}.csv"
        )
        # Write rank-level attempts (Count)
        rank_attempts_csv_name = str(
            output_path + f"RS_simpRAG_rank_attempts_{current_date}_{current_time}.csv"
        )
        write_preds_to_csv(preds, predictions_csv_name)
        write_overall_metrics(final_metrics_csv_name, overalls)
        write_sample_binary_accuracy_csv(preds, sample_binary_csv_name)
        write_rank_attempts_csv(rank_attempts_csv_name, overalls, total_samples=len(preds))

        # Write filtered results (exclude samples with empty predictions) to outputs_uq_data_collection/
        uq_dir = str((Path(output_path).parent / "outputs_uq_data_collection/").as_posix())
        Path(uq_dir).mkdir(parents=True, exist_ok=True)

        from taxonomic_rag_system.utils.helpers import filter_nonempty_results, extract_tax_metrics_rs

        filtered_results = filter_nonempty_results(rarespp_predictions)
        overalls_uq, preds_uq = extract_tax_metrics_rs(filtered_results, verbose=True)

        final_metrics_csv_name_uq = str(
            uq_dir + f"RS_simpRAG_tax_metrics_{current_date}_{current_time}.csv"
        )
        predictions_csv_name_uq = str(
            uq_dir + f"RS_simpRAG_predictions_{current_date}_{current_time}.csv"
        )
        sample_binary_csv_name_uq = str(
            uq_dir + f"RS_simpRAG_sample_binary_accuracy_{current_date}_{current_time}.csv"
        )
        rank_attempts_csv_name_uq = str(
            uq_dir + f"RS_simpRAG_rank_attempts_{current_date}_{current_time}.csv"
        )
        write_preds_to_csv(preds_uq, predictions_csv_name_uq)
        write_overall_metrics(final_metrics_csv_name_uq, overalls_uq)
        write_sample_binary_accuracy_csv(preds_uq, sample_binary_csv_name_uq)
        write_rank_attempts_csv(rank_attempts_csv_name_uq, overalls_uq, total_samples=len(preds_uq))

        # Write filtered JSONL with prompt/response pairs to UQ directory (subset by RSID)
        try:
            filtered_rsids = {s.get("RSID") for s in filtered_results if s.get("RSID")}
            uq_jsonl_name = str(
                uq_dir + f"RS_simpRAG_prompt_response_pairs_{current_date}_{current_time}.jsonl"
            )
            with open(prompt_jsonl_name, "r", encoding="utf-8") as src, open(
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
