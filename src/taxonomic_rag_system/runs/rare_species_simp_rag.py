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
    write_rank_attempts_csv,
    filter_nonempty_results,
    hierarchical_metrics,
    write_per_rank_binary_jsonl,
    sample_hierarchical_metrics,
    per_rank_hierarchical_metrics,
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
    rarespp_predictions = await model.rarespecies_dataset_run(
        interval=interval, verbose=2
    )
    overalls, preds = extract_tax_metrics_rs(rarespp_predictions, verbose=True)

    if write:
        # Create output directory if it doesn't exist
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)

        # Compute hierarchical metrics once and merge into main tax_metrics
        hm = hierarchical_metrics(
            [s.get("true_class", {}) for s in rarespp_predictions],
            [s.get("guess_class", {}) for s in rarespp_predictions],
            verbose=True,
        )
        pr_hm = per_rank_hierarchical_metrics(
            [s.get("true_class", {}) for s in rarespp_predictions],
            [s.get("guess_class", {}) for s in rarespp_predictions],
            verbose=False,
        )
        final_metrics_csv_name = str(
            output_path + f"RS_simpRAG_tax_metrics_hierarchical_{current_date}_{current_time}.csv"
        )
        predictions_csv_name = str(
            output_path + f"RS_simpRAG_predictions_{current_date}_{current_time}.csv"
        )
        # Write per-rank binary accuracy JSONL (one line per rank per sample)
        per_rank_jsonl_name = str(
            output_path + f"RS_simpRAG_per_rank_binary_{current_date}_{current_time}.jsonl"
        )
        # Write rank-level attempts (Count)
        rank_attempts_csv_name = str(
            output_path + f"RS_simpRAG_rank_attempts_{current_date}_{current_time}.csv"
        )
        write_preds_to_csv(preds, predictions_csv_name)
        overalls_with_hier_main = dict(overalls)
        for rank, metrics in overalls_with_hier_main.items():
            if isinstance(metrics, dict) and rank in pr_hm:
                metrics.update(pr_hm[rank])
        overalls_with_hier_main.update(hm)
        write_overall_metrics(final_metrics_csv_name, overalls_with_hier_main)
        write_rank_attempts_csv(rank_attempts_csv_name, overalls, total_samples=len(preds))
        # Write per-rank JSONL using full set
        write_per_rank_binary_jsonl(rarespp_predictions, per_rank_jsonl_name)

        # Augment prompt/response JSONL with per-sample hierarchical metrics (HP/HR/HF)
        try:
            # Build RSID -> metrics map
            rsid_to_hier = {}
            # Also compute BinaryAccuracy per sample (on predicted-only ranks)
            rsid_to_binary = {}
            for s in rarespp_predictions:
                rsid_val = s.get("RSID")
                if rsid_val is None:
                    continue
                m = sample_hierarchical_metrics(s.get("true_class", {}), s.get("guess_class", {}))
                rsid_to_hier[str(rsid_val)] = m
                try:
                    from taxonomic_rag_system.utils.helpers import sample_binary_accuracy
                    ba = sample_binary_accuracy(s.get("true_class", {}) or {}, s.get("guess_class", {}) or {})
                except Exception:
                    ba = None
                rsid_to_binary[str(rsid_val)] = ba
            # Read and rewrite file with augmented response (preserve full response)
            lines_out = []
            with open(prompt_jsonl_name, "r", encoding="utf-8") as src:
                for line in src:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        lines_out.append(line)
                        continue
                    rsid = obj.get("rsid") or obj.get("RSID")
                    if rsid is None:
                        lines_out.append(line)
                        continue
                    hier = rsid_to_hier.get(str(rsid))
                    if hier and isinstance(obj.get("response"), dict):
                        # Preserve original response; attach hierarchical metrics under "hier_metrics"
                        try:
                            # Ensure we don't mutate nested structures unexpectedly
                            resp = dict(obj.get("response") or {})
                        except Exception:
                            resp = obj.get("response") or {}
                        # Add BinaryAccuracy first
                        ba = rsid_to_binary.get(str(rsid))
                        if ba is not None:
                            resp["BinaryAccuracy"] = ba
                        # Add HP/HR/HF into a dedicated field to avoid clobbering richer fields
                        resp["hier_metrics"] = dict(hier)
                        # Ensure lowercase key only
                        if "rsid" not in obj and obj.get("RSID") is not None:
                            obj["rsid"] = obj.get("RSID")
                        obj["response"] = resp
                        line = json.dumps(obj, ensure_ascii=False) + "\n"
                    lines_out.append(line)
            with open(prompt_jsonl_name, "w", encoding="utf-8") as dst:
                for ln in lines_out:
                    dst.write(ln)
        except Exception:
            pass

        # Removed outputs_hierarchical_metrics directory and copies

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
        pr_hm_uq = per_rank_hierarchical_metrics(
            [s.get("true_class", {}) for s in filtered_results],
            [s.get("guess_class", {}) for s in filtered_results],
            verbose=False,
        )

        final_metrics_csv_name_uq = str((uq_dir_path / f"RS_simpRAG_tax_metrics_hierarchical_{current_date}_{current_time}.csv").as_posix())
        predictions_csv_name_uq = str((uq_dir_path / f"RS_simpRAG_predictions_{current_date}_{current_time}.csv").as_posix())
        rank_attempts_csv_name_uq = str((uq_dir_path / f"RS_simpRAG_rank_attempts_{current_date}_{current_time}.csv").as_posix())
        per_rank_jsonl_name_uq = str((uq_dir_path / f"RS_simpRAG_per_rank_binary_{current_date}_{current_time}.jsonl").as_posix())
        write_preds_to_csv(preds_uq, predictions_csv_name_uq)
        overalls_uq_with_hier = dict(overalls_uq)
        for rank, metrics in overalls_uq_with_hier.items():
            if isinstance(metrics, dict) and rank in pr_hm_uq:
                metrics.update(pr_hm_uq[rank])
        overalls_uq_with_hier.update(hm_uq)
        write_overall_metrics(final_metrics_csv_name_uq, overalls_uq_with_hier)
        write_rank_attempts_csv(rank_attempts_csv_name_uq, overalls_uq, total_samples=len(preds_uq))
        # Write per-rank JSONL using filtered set
        write_per_rank_binary_jsonl(filtered_results, per_rank_jsonl_name_uq)
        # Augment filtered JSONL as well by reusing subset copy step below

        # Removed outputs_uq_data_collection_hierarchical_metrics directory and copies

        # Write filtered JSONL with prompt/response pairs to UQ directory (subset by RSID)
        try:
            filtered_rsids = {s.get("RSID") for s in filtered_results if s.get("RSID")}
            uq_jsonl_name = str((uq_dir_path / f"RS_simpRAG_prompt_response_pairs_{current_date}_{current_time}.jsonl").as_posix())
            with open(prompt_jsonl_name, "r", encoding="utf-8") as src, open(
                uq_jsonl_name, "w", encoding="utf-8"
            ) as dst:
                for line in src:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("rsid") in filtered_rsids or obj.get("RSID") in filtered_rsids:
                        dst.write(line)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
