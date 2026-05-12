"""
Caption-only runner using a local Qwen3-VL model (HF Transformers).

This script iterates over the ImageOmics rare-species dataset, generates a detailed
caption per image, and writes results to JSONL as a reusable intermediate artifact.

The output JSONL is designed to be consumed by downstream RAG/LLM stages.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from taxonomic_rag_system.utils.evaluator import RareSpeciesEvaluator
from taxonomic_rag_system.utils.vision_models import Qwen3VLDescriptiveCaptionerLocal


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Path to write caption JSONL (one record per sample).",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Hugging Face model id for the captioner.",
    )
    p.add_argument("--interval-start", type=int, default=0)
    p.add_argument("--interval-end", type=int, default=999)
    p.add_argument("--batch-size", type=int, default=1)

    # Generation / HF placement
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (otherwise uses greedy/beam defaults).",
    )
    p.add_argument("--device-map", type=str, default="cuda0")
    p.add_argument("--torch-dtype", type=str, default="auto")
    p.add_argument("--attn-implementation", type=str, default="sdpa")
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/processor.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If output JSONL exists, skip RSIDs already written.",
    )
    return p.parse_args()


def _load_done_rsids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rsid = obj.get("rsid")
            status = obj.get("status")
            if rsid is None:
                continue
            # Treat any written record (ok or error) as done for resume purposes.
            if status in ("ok", "error"):
                done.add(str(rsid))
    return done


def main() -> None:
    """Run the Qwen3-VL caption loop and append one JSONL record per sample."""
    args = _parse_args()
    out_path = Path(args.output_jsonl).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    interval = (int(args.interval_start), int(args.interval_end))
    print("=== Qwen3-VL caption run ===")
    print(f"model_id: {args.model_id}")
    print(f"output_jsonl: {out_path}")
    print(f"interval: [{interval[0]}, {interval[1]})")
    print(f"batch_size: {args.batch_size}")
    print(
        "gen: "
        f"max_new_tokens={args.max_new_tokens}, do_sample={bool(args.do_sample)}, "
        f"temperature={args.temperature}, top_p={args.top_p}"
    )
    print(
        "hf: "
        f"device_map={args.device_map}, torch_dtype={args.torch_dtype}, "
        f"attn_implementation={args.attn_implementation}, "
        f"trust_remote_code={bool(args.trust_remote_code)}"
    )
    print("============================")

    done_rsids: set[str] = set()
    if bool(args.resume):
        done_rsids = _load_done_rsids(out_path)
        print(f"resume enabled: found {len(done_rsids)} RSIDs already in output JSONL")

    captioner = Qwen3VLDescriptiveCaptionerLocal(
        model_id=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=bool(args.trust_remote_code),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        do_sample=bool(args.do_sample),
    )

    dataloader = RareSpeciesEvaluator(interval=interval).dataloader(
        batch_size=int(args.batch_size)
    )

    # Append mode: safe for resume and for long jobs.
    with out_path.open("a", encoding="utf-8") as f:
        dataset_index = interval[0]
        for image_objs, class_dicts in dataloader:
            for img, meta in zip(image_objs, class_dicts):
                rsid = meta.get("RSID")
                rsid_str = "" if rsid is None else str(rsid)
                if done_rsids and rsid_str in done_rsids:
                    dataset_index += 1
                    continue

                ts = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
                record: dict[str, Any] = {
                    "rsid": rsid_str,
                    "dataset_index": int(dataset_index),
                    "caption": "",
                    "model_id": str(args.model_id),
                    "gen": {
                        "max_new_tokens": int(args.max_new_tokens),
                        "do_sample": bool(args.do_sample),
                        "temperature": float(args.temperature),
                        "top_p": float(args.top_p),
                    },
                    "status": "error",
                    "error": None,
                    "ts": ts,
                }

                try:
                    caption = captioner.generate_caption(img)
                    record["caption"] = caption
                    record["status"] = "ok"
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[caption_error] rsid={rsid_str!r} "
                        f"dataset_index={dataset_index}",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exception(exc, file=sys.stderr, chain=True)

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                dataset_index += 1


if __name__ == "__main__":
    main()

