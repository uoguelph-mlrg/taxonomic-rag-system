"""Generate rare-species captions using an OpenAI-compatible remote VLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
from pathlib import Path
from typing import Any

from taxonomic_rag_system.utils.image_processor import ImageProcessor
from taxonomic_rag_system.utils.remote_vlm import RemoteVLMCaptioner, RemoteVLMConfig


DEFAULT_MODEL = "Qwen2.5-VL-7B-Instruct"


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Caption rare-species images via a remote OpenAI-compatible VLM "
            "(non-empty --base-url required; Slurm stage writes it from "
            "state/vlm_base_url.txt)."
        ),
    )
    parser.add_argument(
        "--server-job-id",
        type=str,
        default="",
        help="Optional Slurm job id of the model server (metadata only).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Required OpenAI-compatible API base URL (e.g. from vlm_base_url.txt).",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--metadata-json", type=str, required=True)
    parser.add_argument("--interval-start", type=int, default=0)
    parser.add_argument("--interval-end", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _require_base_url(base_url: str) -> str:
    """Return stripped OpenAI-compatible endpoint URL; must be non-empty."""
    resolved = (base_url or "").strip()
    if not resolved:
        msg = (
            "Non-empty --base-url is required (e.g. from state/vlm_base_url.txt "
            "after a successful vec-inf Stage 1)."
        )
        raise ValueError(msg)
    return resolved


def _true_class(class_dict: dict[str, Any]) -> dict[str, Any]:
    """Return taxonomy fields without the rare-species identifier."""
    return {k: v for k, v in class_dict.items() if k != "RSID"}


def _load_completed_rsids(output_jsonl: Path) -> set[str]:
    """Read successfully completed RSIDs from an existing JSONL file."""
    completed: set[str] = set()
    if not output_jsonl.exists():
        return completed
    with output_jsonl.open("r", encoding="utf-8") as src:
        for line in src:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rsid = obj.get("rsid")
            if rsid and obj.get("status") == "ok":
                completed.add(str(rsid))
    return completed


def _build_dataloader(
    *,
    interval_start: int,
    interval_end: int,
    batch_size: int,
) -> Any:
    """Build the rare-species dataloader lazily to keep imports test-friendly."""
    from taxonomic_rag_system.utils.evaluator import RareSpeciesEvaluator

    return RareSpeciesEvaluator(
        interval=(interval_start, interval_end),
    ).dataloader(batch_size=batch_size)


async def _caption_one(
    *,
    captioner: RemoteVLMCaptioner,
    image_obj: Any,
    class_dict: dict[str, Any],
    model: str,
    base_url: str,
    interval_start: int,
    interval_end: int,
) -> dict[str, Any]:
    """Caption one rare-species sample and return a JSON-serializable record."""
    rsid = str(class_dict.get("RSID", ""))
    record: dict[str, Any] = {
        "rsid": rsid,
        "caption": "",
        "true_class": _true_class(class_dict),
        "model": model,
        "base_url": base_url,
        "status": "error",
        "error": None,
        "interval_start": interval_start,
        "interval_end": interval_end,
        "created_at": _utc_now(),
    }
    try:
        image_b64 = ImageProcessor.process_image(image_obj=image_obj)
        caption = await captioner.generate_caption(image_b64)
        record["caption"] = caption
        if caption:
            record["status"] = "ok"
        else:
            record["error"] = "empty caption"
    except Exception as e:
        record["error"] = str(e)
    return record


async def run_captioning(args: argparse.Namespace) -> dict[str, Any]:
    """Run remote VLM captioning and write JSONL plus run metadata."""
    output_jsonl = Path(args.output_jsonl)
    metadata_json = Path(args.metadata_json)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    base_url = _require_base_url(args.base_url)
    metadata: dict[str, Any] = {
        "model": args.model,
        "server_job_id": args.server_job_id,
        "base_url": base_url,
        "interval_start": args.interval_start,
        "interval_end": args.interval_end,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "output_jsonl": str(output_jsonl),
        "started_at": started_at,
        "completed_at": None,
        "records_total": 0,
        "records_ok": 0,
        "records_error": 0,
        "resume": bool(args.resume),
    }
    metadata_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    completed_rsids = _load_completed_rsids(output_jsonl) if args.resume else set()
    mode = "a" if args.resume else "w"
    captioner = RemoteVLMCaptioner(
        RemoteVLMConfig(
            model=args.model,
            base_url=base_url,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )
    dataloader = _build_dataloader(
        interval_start=args.interval_start,
        interval_end=args.interval_end,
        batch_size=args.batch_size,
    )

    with output_jsonl.open(mode, encoding="utf-8") as dst:
        for image_objs, class_dicts in dataloader:
            tasks = []
            for image_obj, class_dict in zip(image_objs, class_dicts):
                rsid = str(class_dict.get("RSID", ""))
                if rsid in completed_rsids:
                    continue
                tasks.append(
                    _caption_one(
                        captioner=captioner,
                        image_obj=image_obj,
                        class_dict=class_dict,
                        model=args.model,
                        base_url=base_url,
                        interval_start=args.interval_start,
                        interval_end=args.interval_end,
                    ),
                )
            if not tasks:
                continue
            records = await asyncio.gather(*tasks)
            for record in records:
                metadata["records_total"] += 1
                if record["status"] == "ok":
                    metadata["records_ok"] += 1
                else:
                    metadata["records_error"] += 1
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            dst.flush()

    metadata["completed_at"] = _utc_now()
    metadata_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    """CLI entry point."""
    args = _parse_arguments()
    metadata = asyncio.run(run_captioning(args))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
