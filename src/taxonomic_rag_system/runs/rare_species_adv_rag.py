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
    main: Executes Advanced RAG run, extracts metrics, and optionally writes results.

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
    filter_nonempty_results,
    hierarchical_metrics,
    write_sample_hierarchical_metrics_csv,
    write_preds_hierarchical_to_csv,
    write_per_rank_binary_jsonl,
    write_sample_binary_hierarchical_json,
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
    interval = (args.interval_start, args.interval_end)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    prompt_jsonl_name = str(
        output_path + f"RS_advRAG_prompt_response_pairs_{current_date}_{current_time}.jsonl"
    )

    # 1. Build Model
    model = ImageRAGModel(
        vstore_path=vstore_path,
        model="gpt-4o",
        search_type="mmr",
        k=10,
        rerank=True,
        multiquery=True,
        prompt_log_path=prompt_jsonl_name if write else None,
    )
    print(f"Device: {model.get_device()}")
    # 2. RareSpecies Run
    rarespp_predictions = await model.rarespecies_dataset_run(
        interval=interval, verbose=2
    )
    # 3. Extract Results
    overalls, preds = extract_tax_metrics_rs(rarespp_predictions, verbose=True)

    if write:
        # Create output directory if it doesn't exist
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)

        # Merge hierarchical metrics into main tax_metrics and rename to include hierarchical
        hm = hierarchical_metrics(
            [s.get("true_class", {}) for s in rarespp_predictions],
            [s.get("guess_class", {}) for s in rarespp_predictions],
            verbose=True,
        )
        final_metrics_csv_name = str(
            output_path + f"RS_advRAG_tax_metrics_hierarchical_{current_date}_{current_time}.csv"
        )
        predictions_csv_name = str(
            output_path + f"RS_advRAG_predictions_{current_date}_{current_time}.csv"
        )
        # Per-rank binary accuracy JSONL
        per_rank_jsonl_name = str(
            output_path + f"RS_advRAG_per_rank_binary_{current_date}_{current_time}.jsonl"
        )
        # Per-sample binary + hierarchical JSON
        per_sample_metrics_json = str(
            output_path + f"RS_advRAG_per_sample_binary_hierarchical_{current_date}_{current_time}.json"
        )
        # Write per-sample binary accuracy labels
        sample_binary_csv_name = str(
            output_path + f"RS_advRAG_sample_binary_accuracy_{current_date}_{current_time}.csv"
        )
        # Write rank-level attempts (Count)
        rank_attempts_csv_name = str(
            output_path + f"RS_advRAG_rank_attempts_{current_date}_{current_time}.csv"
        )
        write_preds_to_csv(preds, predictions_csv_name)
        overalls_with_hier_main = dict(overalls)
        overalls_with_hier_main.update(hm)
        write_overall_metrics(final_metrics_csv_name, overalls_with_hier_main)
        write_sample_binary_accuracy_csv(preds, sample_binary_csv_name)
        write_rank_attempts_csv(rank_attempts_csv_name, overalls, total_samples=len(preds))
        # Write per-rank JSONL (full set)
        write_per_rank_binary_jsonl(rarespp_predictions, per_rank_jsonl_name)
        # Write per-sample JSON (binary + hierarchical)
        write_sample_binary_hierarchical_json(rarespp_predictions, per_sample_metrics_json)

        # Hierarchical metrics outputs
        hier_dir_path = Path(output_path).parent / "outputs_hierarchical_metrics"
        hier_dir_path.mkdir(parents=True, exist_ok=True)
        # `hm` computed above
        # Write hierarchical tax metrics and per-sample hierarchical CSVs
        hier_tax_metrics_csv = str((hier_dir_path / f"RS_advRAG_tax_metrics_{current_date}_{current_time}.csv").as_posix())
        overalls_with_hier = dict(overalls)
        overalls_with_hier.update(hm)
        write_overall_metrics(hier_tax_metrics_csv, overalls_with_hier)
        hier_predictions_csv = str((hier_dir_path / f"RS_advRAG_predictions_{current_date}_{current_time}.csv").as_posix())
        write_preds_hierarchical_to_csv(rarespp_predictions, hier_predictions_csv)
        hier_sample_csv = str((hier_dir_path / f"RS_advRAG_sample_hierarchical_metrics_{current_date}_{current_time}.csv").as_posix())
        write_sample_hierarchical_metrics_csv(rarespp_predictions, hier_sample_csv)
        import shutil as _shutil
        _shutil.copy2(rank_attempts_csv_name, str((hier_dir_path / f"RS_advRAG_rank_attempts_{current_date}_{current_time}.csv").as_posix()))
        _shutil.copy2(prompt_jsonl_name, str((hier_dir_path / f"RS_advRAG_prompt_response_pairs_{current_date}_{current_time}.jsonl").as_posix()))
        _shutil.copy2(per_rank_jsonl_name, str((hier_dir_path / f"RS_advRAG_per_rank_binary_{current_date}_{current_time}.jsonl").as_posix()))
        _shutil.copy2(per_sample_metrics_json, str((hier_dir_path / f"RS_advRAG_per_sample_binary_hierarchical_{current_date}_{current_time}.json").as_posix()))

        # Write filtered results (exclude samples with empty predictions) to outputs_uq_data_collection/
        uq_dir_path = Path(output_path).parent / "outputs_uq_data_collection"
        uq_dir_path.mkdir(parents=True, exist_ok=True)

        filtered_results = filter_nonempty_results(rarespp_predictions)
        overalls_uq, preds_uq = extract_tax_metrics_rs(filtered_results, verbose=True)
        hm_uq = hierarchical_metrics(
            [s.get("true_class", {}) for s in filtered_results],
            [s.get("guess_class", {}) for s in filtered_results],
            verbose=True,
        )

        final_metrics_csv_name_uq = str((uq_dir_path / f"RS_advRAG_tax_metrics_hierarchical_{current_date}_{current_time}.csv").as_posix())
        predictions_csv_name_uq = str((uq_dir_path / f"RS_advRAG_predictions_{current_date}_{current_time}.csv").as_posix())
        sample_binary_csv_name_uq = str((uq_dir_path / f"RS_advRAG_sample_binary_accuracy_{current_date}_{current_time}.csv").as_posix())
        rank_attempts_csv_name_uq = str((uq_dir_path / f"RS_advRAG_rank_attempts_{current_date}_{current_time}.csv").as_posix())
        per_rank_jsonl_name_uq = str((uq_dir_path / f"RS_advRAG_per_rank_binary_{current_date}_{current_time}.jsonl").as_posix())
        per_sample_metrics_json_uq = str((uq_dir_path / f"RS_advRAG_per_sample_binary_hierarchical_{current_date}_{current_time}.json").as_posix())
        write_preds_to_csv(preds_uq, predictions_csv_name_uq)
        overalls_uq_with_hier = dict(overalls_uq)
        overalls_uq_with_hier.update(hm_uq)
        write_overall_metrics(final_metrics_csv_name_uq, overalls_uq_with_hier)
        write_sample_binary_accuracy_csv(preds_uq, sample_binary_csv_name_uq)
        write_rank_attempts_csv(rank_attempts_csv_name_uq, overalls_uq, total_samples=len(preds_uq))
        # Per-rank JSONL for filtered subset
        write_per_rank_binary_jsonl(filtered_results, per_rank_jsonl_name_uq)
        # Per-sample JSON for filtered subset
        write_sample_binary_hierarchical_json(filtered_results, per_sample_metrics_json_uq)

        # Removed outputs_uq_data_collection_hierarchical_metrics directory and copies

        # Write filtered JSONL with prompt/response pairs to UQ directory (subset by RSID)
        try:
            filtered_rsids = {s.get("RSID") for s in filtered_results if s.get("RSID")}
            uq_jsonl_name = str((uq_dir_path / f"RS_advRAG_prompt_response_pairs_{current_date}_{current_time}.jsonl").as_posix())
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
