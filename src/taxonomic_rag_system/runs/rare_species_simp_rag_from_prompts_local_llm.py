"""
Run a local open-source LLM on stored SimpleRAG prompts.

This runner bypasses the VLM and retriever stages by reading an existing
`RS_simpRAG_prompt_response_pairs_*.jsonl` file and feeding the stored `prompt`
directly into a local Hugging Face model (e.g., Qwen3).

Outputs are written in the same schemas as the SimpleRAG pipeline outputs,
excluding logprobs:
  - RS_simpRAG_prompt_response_pairs_*.jsonl
  - RS_simpRAG_predictions_*.csv
  - RS_simpRAG_per_rank_binary_*.jsonl
  - RS_simpRAG_tax_metrics_hierarchical_*.csv
  - RS_simpRAG_rank_attempts_*.csv
"""

from __future__ import annotations

import argparse
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
from taxonomic_rag_system.utils.local_llm import LocalLLMConfig, build_hf_textgen_llm
from taxonomic_rag_system.utils.out_models import TaxBiodiversity
from taxonomic_rag_system.utils.rare_species_gold import build_rsid_to_true_class


REQUIRED_RESPONSE_KEYS = {
    "classification",
    "ancestral",
    "specific",
    "commentary",
    "bio_knowledge",
}
JSON_ONLY_RETRY_SUFFIX = "\n\nOutput ONLY the JSON object. Do not use markdown.\n"


def _fmt_optional(value: Any) -> str:
    """Format optional CLI values for startup logging."""
    return "<model default>" if value is None else str(value)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to RS_simpRAG_prompt_response_pairs_*.jsonl containing stored prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output directory prefix (same as other runners).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write outputs to disk.",
    )
    parser.add_argument(
        "--interval-start",
        type=int,
        default=0,
        help="Start line index (0-based) in input JSONL (default: 0).",
    )
    parser.add_argument(
        "--interval-end",
        type=int,
        default=10**9,
        help="End line index (exclusive) in input JSONL (default: all).",
    )

    # Local HF model options
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model id or local path for Qwen3.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the model default max_new_tokens. Omit to use the model default.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the model default temperature. Omit to use the model default.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override the model default top_p. Omit to use the model default.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda0",
        help=(
            "HF device_map: 'auto', JSON object (e.g. '{\"\": 0}'), or single-GPU alias "
            'cuda0/gpu0/single/0/cuda:0 (maps to {"": 0} to avoid CPU offload mixups).'
        ),
    )
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model.",
    )
    return parser.parse_args()


def _extract_outer_json_object(text: str) -> str:
    """Best-effort extraction of the outermost JSON object from model output."""
    s = text.strip()
    # Strip common markdown fences
    if s.startswith("```"):
        parts = s.split("```")
        # Try to keep the largest fenced block content
        if len(parts) >= 3:
            s = parts[1].strip()
    # Find outer braces
    lb = s.find("{")
    rb = s.rfind("}")
    if lb != -1 and rb != -1 and rb > lb:
        return s[lb : rb + 1]
    return s


def _coerce_taxbiodiversity_dict(obj: Any) -> dict[str, Any]:
    """Validate and coerce a decoded JSON object into the TaxBiodiversity schema."""
    if not isinstance(obj, dict):
        raise ValueError("decoded_json_not_object")
    missing = REQUIRED_RESPONSE_KEYS - set(obj.keys())
    if missing:
        raise ValueError(f"missing_keys:{sorted(missing)}")
    tb = TaxBiodiversity.model_validate(obj)
    return tb.model_dump()


def _invoke_structured_taxonomy_response(llm: Any, prompt: str) -> dict[str, Any]:
    """Run the local LLM and require a schema-valid JSON response."""
    errors: list[str] = []

    for attempt in range(2):
        prompt_to_use = prompt
        if attempt == 1:
            prompt_to_use = prompt.rstrip() + JSON_ONLY_RETRY_SUFFIX
        try:
            raw_text = llm.invoke(prompt_to_use)
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)
            cleaned = _extract_outer_json_object(raw_text)
            decoded = json.loads(cleaned)
            return _coerce_taxbiodiversity_dict(decoded)
        except Exception as exc:
            errors.append(f"attempt {attempt + 1}: {exc}")

    detail = " | ".join(errors) if errors else "unknown error"
    raise RuntimeError(
        "Local LLM inference failed after 2 attempts; refusing to emit fallback "
        f"taxonomy. {detail}"
    )


def main() -> None:  # noqa: PLR0912, PLR0915
    """Load prompts, run local HF LLM, write metrics and prediction artifacts."""
    args = _parse_arguments()
    input_jsonl = args.input_jsonl
    write = args.write
    output_path = args.output
    interval = (args.interval_start, args.interval_end)

    print("=== Local prompt->LLM run configuration ===")
    print(f"model_id: {args.model_id}")
    print(f"device_map: {args.device_map}")
    print(f"torch_dtype: {args.torch_dtype}")
    print(f"trust_remote_code: {bool(args.trust_remote_code)}")
    print(f"max_new_tokens: {_fmt_optional(args.max_new_tokens)}")
    print(f"temperature: {_fmt_optional(args.temperature)}")
    print(f"top_p: {_fmt_optional(args.top_p)}")
    generation_overrides = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    active_overrides = [
        name for name, value in generation_overrides.items() if value is not None
    ]
    if active_overrides:
        print(f"generation overrides active: {', '.join(active_overrides)}")
    else:
        print("generation overrides omitted -> use model generation_config defaults")
    if args.top_p is not None and args.temperature is None:
        print(
            "WARNING: --top-p was provided without --temperature; top_p will be ignored "
            "because sampling overrides are not enabled."
        )
    print("==========================================")

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")

    prompt_jsonl_name = str(
        output_path
        + f"RS_simpRAG_prompt_response_pairs_{current_date}_{current_time}.jsonl"
    )
    metrics_csv_name = str(
        output_path
        + f"RS_simpRAG_tax_metrics_hierarchical_{current_date}_{current_time}.csv"
    )
    predictions_csv_name = str(
        output_path + f"RS_simpRAG_predictions_{current_date}_{current_time}.csv"
    )
    per_rank_jsonl_name = str(
        output_path + f"RS_simpRAG_per_rank_binary_{current_date}_{current_time}.jsonl"
    )
    rank_attempts_csv_name = str(
        output_path + f"RS_simpRAG_rank_attempts_{current_date}_{current_time}.csv"
    )

    if write and output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    # Read input prompts
    rows: list[dict[str, Any]] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < interval[0]:
                continue
            if i >= interval[1]:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rsid = obj.get("rsid")
            prompt = obj.get("prompt")
            if rsid is None or prompt is None:
                continue
            rows.append({"rsid": str(rsid), "prompt": str(prompt)})

    print(
        "Loaded "
        f"{len(rows)} prompts from {input_jsonl} for interval [{interval[0]}, {interval[1]})"
    )
    if not rows:
        raise SystemExit("No valid prompt rows were loaded from the input JSONL.")

    rsids = {r["rsid"] for r in rows}
    gold_map = build_rsid_to_true_class(rsids)
    print(f"Loaded {len(gold_map)} gold taxonomy records")

    llm = build_hf_textgen_llm(
        cfg=LocalLLMConfig(
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            trust_remote_code=bool(args.trust_remote_code),
            return_full_text=False,
        )
    )
    print(f"Writing outputs under: {output_path or './'}")

    out_jsonl_lines: list[str] = []
    samples: list[dict[str, Any]] = []

    for r in rows:
        rsid = r["rsid"]
        prompt = r["prompt"]
        try:
            resp_dict = _invoke_structured_taxonomy_response(llm, prompt)
        except Exception as exc:
            raise RuntimeError(f"Failed for RSID {rsid}: {exc}") from exc

        out_jsonl_lines.append(
            json.dumps(
                {"rsid": rsid, "prompt": prompt, "response": resp_dict},
                ensure_ascii=False,
            )
            + "\n"
        )

        true_class = gold_map.get(rsid, {})
        # guess_class: drop "N/A" to match existing pipeline behaviour
        cls = (
            (resp_dict.get("classification") or {})
            if isinstance(resp_dict, dict)
            else {}
        )
        guess_class = {
            k: v for (k, v) in cls.items() if isinstance(v, str) and v != "N/A"
        }
        guess_class = {k: v for (k, v) in guess_class.items() if k != "Domain"}

        samples.append(
            {
                "RSID": rsid,
                "true_class": true_class,
                "guess_class": guess_class,
            }
        )

    # Write prompt/response JSONL
    if write:
        with open(prompt_jsonl_name, "w", encoding="utf-8") as f:
            for line in out_jsonl_lines:
                f.write(line)

    # Produce legacy outputs (metrics/preds/per-rank)
    overalls, preds = extract_tax_metrics_rs(samples, verbose=True)

    if write:
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

        write_overall_metrics(metrics_csv_name, overalls_with_hier)
        write_rank_attempts_csv(
            rank_attempts_csv_name, overalls, total_samples=len(preds)
        )


if __name__ == "__main__":
    main()
