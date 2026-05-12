"""
Run the Simple RAG model starting from stored captions JSONL.

This runner bypasses the VLM stage and uses a caption JSONL file produced by:
  - `rare_species_caption_qwen3vl_local.py`

For each caption, it runs retrieval + LLM generation and writes the same set of
outputs as the standard SimpleRAG runner (predictions + metrics). Optionally,
it can also log prompt/response pairs via the existing prompt logger.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
from pathlib import Path
from typing import Any

from taxonomic_rag_system.utils.helpers import (
    extract_tax_metrics_rs,
    hierarchical_metrics,
    per_rank_hierarchical_metrics,
    write_overall_metrics,
    write_per_rank_binary_jsonl,
    write_preds_to_csv,
    write_rank_attempts_csv,
)
from taxonomic_rag_system.utils.retriever import WikiStellaRAGModel
from taxonomic_rag_system.utils.rare_species_gold import build_rsid_to_true_class


def _parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--captions-jsonl", type=str, required=True)
    p.add_argument("--vstore", type=str, required=True)
    p.add_argument("--write", action="store_true")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--interval-start", type=int, default=0)
    p.add_argument("--interval-end", type=int, default=10**9)

    # Retriever knobs
    p.add_argument("--collection-name", type=str, default="Wiki_contexted")
    p.add_argument("--embedding-model", type=str, default="dunzhang/stella_en_1.5B_v5")
    p.add_argument("--search-type", type=str, default="similarity")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--multiquery", action="store_true")

    return p.parse_args()


def _read_caption_rows(
    path: Path, interval_start: int, interval_end: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < interval_start:
                continue
            if i >= interval_end:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("status") != "ok":
                continue
            rsid = obj.get("rsid")
            caption = obj.get("caption")
            if rsid is None or caption is None:
                continue
            rows.append({"rsid": str(rsid), "caption": str(caption)})
    return rows


async def main() -> None:
    args = _parse_arguments()
    out_dir = Path(args.output).expanduser() if args.output else Path("")
    if args.write and args.output:
        out_dir.mkdir(parents=True, exist_ok=True)

    interval = (int(args.interval_start), int(args.interval_end))
    captions_path = Path(args.captions_jsonl).expanduser()
    rows = _read_caption_rows(captions_path, interval[0], interval[1])
    if not rows:
        raise SystemExit("No usable caption rows were loaded from the input JSONL.")
    rsids = {r["rsid"] for r in rows}
    gold_map = build_rsid_to_true_class(rsids)
    print(f"Loaded {len(gold_map)} gold taxonomy records")
    rows = [r for r in rows if r["rsid"] in gold_map]
    if not rows:
        raise SystemExit("No caption rows remained after gold-label filtering.")

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    output_prefix = args.output or ""

    prompt_jsonl_name = str(
        output_prefix + f"RS_simpRAG_prompt_response_pairs_{current_date}_{current_time}.jsonl"
    )
    logprobs_jsonl_name = str(
        output_prefix + f"RS_simpRAG_logprobs_{current_date}_{current_time}.jsonl"
    )

    rag = WikiStellaRAGModel(
        vstore_path=args.vstore,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        search_type=args.search_type,
        k=int(args.k),
        rerank=bool(args.rerank),
        multiquery=bool(args.multiquery),
        log_path=prompt_jsonl_name if args.write else None,
        logprobs_path=logprobs_jsonl_name if args.write else None,
    )

    samples: list[dict[str, Any]] = []
    for r in rows:
        rsid = r["rsid"]
        caption = r["caption"]
        resp = await rag.ainvoke(caption=caption, RSID=rsid)
        cls = resp.classification or {}
        guess_class = {k: v for (k, v) in cls.items() if isinstance(v, str) and v != "N/A"}
        guess_class = {k: v for (k, v) in guess_class.items() if k != "Domain"}
        true_class = gold_map.get(rsid, {})
        samples.append({"RSID": rsid, "true_class": true_class, "guess_class": guess_class})

    # If caller wants full evaluation, they can merge gold later; for now we still
    # emit the legacy outputs with empty gold (metrics will be empty/NaN-ish).
    overalls, preds = extract_tax_metrics_rs(samples, verbose=True)

    if args.write:
        predictions_csv_name = str(
            output_prefix + f"RS_simpRAG_predictions_{current_date}_{current_time}.csv"
        )
        final_metrics_csv_name = str(
            output_prefix
            + f"RS_simpRAG_tax_metrics_hierarchical_{current_date}_{current_time}.csv"
        )
        per_rank_jsonl_name = str(
            output_prefix + f"RS_simpRAG_per_rank_binary_{current_date}_{current_time}.jsonl"
        )
        rank_attempts_csv_name = str(
            output_prefix + f"RS_simpRAG_rank_attempts_{current_date}_{current_time}.csv"
        )

        write_preds_to_csv(preds, predictions_csv_name)
        write_per_rank_binary_jsonl(samples, per_rank_jsonl_name)

        hm = hierarchical_metrics(
            [s.get("true_class", {}) for s in samples],
            [s.get("guess_class", {}) for s in samples],
            verbose=True,
        )
        pr_hm = per_rank_hierarchical_metrics(
            [s.get("true_class", {}) for s in samples],
            [s.get("guess_class", {}) for s in samples],
            verbose=False,
        )
        overalls_with_hier = dict(overalls)
        for rank, metrics in overalls_with_hier.items():
            if isinstance(metrics, dict) and rank in pr_hm:
                metrics.update(pr_hm[rank])
        overalls_with_hier.update(hm)
        write_overall_metrics(final_metrics_csv_name, overalls_with_hier)
        write_rank_attempts_csv(rank_attempts_csv_name, overalls, total_samples=len(preds))


if __name__ == "__main__":
    asyncio.run(main())

