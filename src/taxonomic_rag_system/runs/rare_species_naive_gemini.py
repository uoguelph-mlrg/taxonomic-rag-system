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
import json
from pathlib import Path

from taxonomic_rag_system.core.image_rag import NaiveVLModel
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
    prompt_jsonl_name = str(
        (Path(output_path) / f"RS_naiveVLM_gemini_prompt_response_pairs_{current_date}_{current_time}.jsonl").as_posix()
    )

    # 1. Build Model, 2. RareSpecies Run, 3. Extract Results
    naive_model = NaiveVLModel(model="google/gemini-2.0-flash-001")
    rarespp_predictions = await naive_model.rarespecies_dataset_run(
        interval=interval, verbose=2
    )
    overalls, preds = extract_tax_metrics_rs(rarespp_predictions, verbose=True)

    if write:
        # Create output directory if it doesn't exist
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)

        # Merge hierarchical metrics into main tax_metrics
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
            output_path
            + f"RS_naiveVLM_gemini_tax_metrics_hierarchical_{current_date}_{current_time}.csv"
        )
        predictions_csv_name = str(
            output_path
            + f"RS_naiveVLM_gemini_predictions_{current_date}_{current_time}.csv"
        )
        # Per-rank binary accuracy JSONL
        per_rank_jsonl_name = str(
            output_path + f"RS_naiveVLM_gemini_per_rank_binary_{current_date}_{current_time}.jsonl"
        )
        # (Removed) Per-sample JSON and sample-binary CSV
        # Write rank-level attempts (Count)
        rank_attempts_csv_name = str(
            output_path
            + f"RS_naiveVLM_gemini_rank_attempts_{current_date}_{current_time}.csv"
        )
        write_preds_to_csv(preds, predictions_csv_name)
        overalls_with_hier_main = dict(overalls)
        for rank, metrics in overalls_with_hier_main.items():
            if isinstance(metrics, dict) and rank in pr_hm:
                metrics.update(pr_hm[rank])
        overalls_with_hier_main.update(hm)
        write_overall_metrics(final_metrics_csv_name, overalls_with_hier_main)
        write_rank_attempts_csv(rank_attempts_csv_name, overalls, total_samples=len(preds))
        # Write per-rank JSONL (full set)
        write_per_rank_binary_jsonl(rarespp_predictions, per_rank_jsonl_name)
        # Write prompt/response JSONL for Gemini naive model
        try:
            prompt_text = getattr(getattr(naive_model, "model", object()), "system_prompt", "")
            temp_val = getattr(getattr(naive_model, "model", object()), "temp", None)
            target_params = {
                "temperature": temp_val if temp_val is not None else "N/A",
                "top_p": "N/A",
            }
            with open(prompt_jsonl_name, "w", encoding="utf-8") as dst:
                for sample in rarespp_predictions:
                    try:
                        rsid_val = sample.get("RSID")
                        gc = sample.get("guess_class") or {}
                        response_obj = {"classification": gc}
                        json_line = json.dumps(
                            {
                                "rsid": rsid_val,
                                "prompt": prompt_text,
                                "response": response_obj,
                                "targetLLM_params": target_params,
                            },
                            ensure_ascii=False,
                        )
                    except Exception:
                        json_line = json.dumps({}, ensure_ascii=False)
                    dst.write(json_line + "\n")
        except Exception:
            pass

        # Removed outputs_hierarchical_metrics directory and copies

        # Write filtered results (exclude samples with empty predictions) to outputs_uq_data_collection/
        uq_dir_path = Path(output_path).parent / "outputs_uq_data_collection"
        uq_dir_path.mkdir(parents=True, exist_ok=True)

        filtered = filter_nonempty_results(rarespp_predictions)
        overalls_uq, preds_uq = extract_tax_metrics_rs(filtered, verbose=True)
        # Enrich preds_uq with RSID from filtered_results for downstream CSV writers
        rsids_uq = [s.get("RSID") for s in filtered]
        for g, r in zip(preds_uq, rsids_uq):
            g["RSID"] = r

        hm_uq = hierarchical_metrics(
            [s.get("true_class", {}) for s in filtered],
            [s.get("guess_class", {}) for s in filtered],
            verbose=True,
        )
        pr_hm_uq = per_rank_hierarchical_metrics(
            [s.get("true_class", {}) for s in filtered],
            [s.get("guess_class", {}) for s in filtered],
            verbose=False,
        )
        final_metrics_csv_name_uq = str(
            (uq_dir_path / f"RS_naiveVLM_gemini_tax_metrics_hierarchical_{current_date}_{current_time}.csv").as_posix()
        )
        predictions_csv_name_uq = str(
            (uq_dir_path / f"RS_naiveVLM_gemini_predictions_{current_date}_{current_time}.csv").as_posix()
        )
        rank_attempts_csv_name_uq = str(
            (uq_dir_path / f"RS_naiveVLM_gemini_rank_attempts_{current_date}_{current_time}.csv").as_posix()
        )
        per_rank_jsonl_name_uq = str(
            (uq_dir_path / f"RS_naiveVLM_gemini_per_rank_binary_{current_date}_{current_time}.jsonl").as_posix()
        )
        write_preds_to_csv(preds_uq, predictions_csv_name_uq)
        overalls_uq_with_hier = dict(overalls_uq)
        for rank, metrics in overalls_uq_with_hier.items():
            if isinstance(metrics, dict) and rank in pr_hm_uq:
                metrics.update(pr_hm_uq[rank])
        overalls_uq_with_hier.update(hm_uq)
        write_overall_metrics(final_metrics_csv_name_uq, overalls_uq_with_hier)
        write_rank_attempts_csv(rank_attempts_csv_name_uq, overalls_uq, total_samples=len(preds_uq))
        # Write per-rank JSONL (filtered)
        write_per_rank_binary_jsonl(filtered, per_rank_jsonl_name_uq)

        # Removed outputs_uq_data_collection_hierarchical_metrics directory and associated copies

        # Write filtered JSONL with prompt/response pairs to UQ directory (subset by RSID)
        try:
            filtered_rsids = {s.get("RSID") for s in filtered if s.get("RSID")}
            uq_jsonl_name = str((uq_dir_path / f"RS_naiveVLM_gemini_prompt_response_pairs_{current_date}_{current_time}.jsonl").as_posix())
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
