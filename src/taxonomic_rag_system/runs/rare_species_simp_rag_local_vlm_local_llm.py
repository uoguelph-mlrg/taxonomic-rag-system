"""Run the SimpleRAG pipeline with a local VLM and a local HF LLM."""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
from pathlib import Path

from taxonomic_rag_system.core.image_rag import ImageRAGModel
from taxonomic_rag_system.utils.helpers import (
    extract_tax_metrics_rs,
    hierarchical_metrics,
    per_rank_hierarchical_metrics,
    sample_binary_accuracy,
    sample_hierarchical_metrics,
    write_overall_metrics,
    write_per_rank_binary_jsonl,
    write_preds_to_csv,
    write_rank_attempts_csv,
)
from taxonomic_rag_system.utils.local_llm import LocalLLMConfig, build_hf_textgen_llm
from taxonomic_rag_system.utils.local_vlm import LocalVLMCaptioner, LocalVLMConfig


DEFAULT_LLM_CHOICES: dict[str, dict[str, str | bool | int]] = {
    "qwen": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "trust_remote_code": True,
    },
    "gptoss": {
        "model_id": "openai/gpt-oss-20b",
        "trust_remote_code": False,
    },
    "gemma": {
        "model_id": "google/gemma-7b-it",
        "trust_remote_code": False,
        "max_new_tokens": 512,
    },
}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vstore", type=str, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--interval-start", type=int, default=0)
    parser.add_argument("--interval-end", type=int, default=999)

    parser.add_argument(
        "--vlm-model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--vlm-max-new-tokens", type=int, default=1024)
    parser.add_argument("--vlm-device-map", type=str, default="cuda0")
    parser.add_argument("--vlm-torch-dtype", type=str, default="auto")
    parser.add_argument("--vlm-trust-remote-code", action="store_true")

    parser.add_argument(
        "--model-choice",
        choices=sorted(DEFAULT_LLM_CHOICES),
        default="gptoss",
        help="High-level alias for the local text-generation model.",
    )
    parser.add_argument(
        "--llm-model-id",
        type=str,
        default="",
        help="Optional explicit HF model id overriding --model-choice.",
    )
    parser.add_argument("--llm-max-new-tokens", type=int, default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--llm-top-p", type=float, default=None)
    parser.add_argument("--llm-device-map", type=str, default="cuda0")
    parser.add_argument("--llm-torch-dtype", type=str, default="auto")
    parser.add_argument("--llm-trust-remote-code", action="store_true")
    return parser.parse_args()


def _resolve_llm_choice(args: argparse.Namespace) -> tuple[str, bool, int | None]:
    defaults = DEFAULT_LLM_CHOICES[args.model_choice]
    model_id = args.llm_model_id or str(defaults["model_id"])
    trust_remote_code = (
        bool(args.llm_trust_remote_code)
        if args.llm_trust_remote_code
        else bool(defaults["trust_remote_code"])
    )
    max_new_tokens = args.llm_max_new_tokens
    if max_new_tokens is None and "max_new_tokens" in defaults:
        max_new_tokens = int(defaults["max_new_tokens"])
    return model_id, trust_remote_code, max_new_tokens


def _build_local_llm(args: argparse.Namespace):
    model_id, trust_remote_code, max_new_tokens = _resolve_llm_choice(args)
    cfg = LocalLLMConfig(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
        device_map=args.llm_device_map,
        torch_dtype=args.llm_torch_dtype,
        trust_remote_code=trust_remote_code,
        return_full_text=False,
    )
    return build_hf_textgen_llm(cfg=cfg)


async def main() -> None:  # noqa: PLR0912, PLR0915
    """Run the local-VLM + retrieval + local-LLM pipeline end to end."""
    args = _parse_arguments()
    interval = (args.interval_start, args.interval_end)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")

    prompt_jsonl_name = str(
        args.output + f"RS_simpRAG_prompt_response_pairs_{current_date}_{current_time}.jsonl"
    )
    metrics_csv_name = str(
        args.output + f"RS_simpRAG_tax_metrics_hierarchical_{current_date}_{current_time}.csv"
    )
    predictions_csv_name = str(
        args.output + f"RS_simpRAG_predictions_{current_date}_{current_time}.csv"
    )
    per_rank_jsonl_name = str(
        args.output + f"RS_simpRAG_per_rank_binary_{current_date}_{current_time}.jsonl"
    )
    rank_attempts_csv_name = str(
        args.output + f"RS_simpRAG_rank_attempts_{current_date}_{current_time}.csv"
    )

    if args.write and args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    llm_model_id, llm_trust_remote_code, llm_max_new_tokens = _resolve_llm_choice(args)
    print("=== Local end-to-end SimpleRAG configuration ===")
    print(f"vlm_model_id: {args.vlm_model_id}")
    print(f"vlm_device_map: {args.vlm_device_map}")
    print(f"llm_choice: {args.model_choice}")
    print(f"llm_model_id: {llm_model_id}")
    print(f"llm_device_map: {args.llm_device_map}")
    print(f"llm_torch_dtype: {args.llm_torch_dtype}")
    print(f"llm_trust_remote_code: {llm_trust_remote_code}")
    print(f"llm_max_new_tokens: {llm_max_new_tokens}")
    print("===============================================")

    captioner = LocalVLMCaptioner(
        LocalVLMConfig(
            model_id=args.vlm_model_id,
            max_new_tokens=args.vlm_max_new_tokens,
            device_map=args.vlm_device_map,
            torch_dtype=args.vlm_torch_dtype,
            trust_remote_code=bool(args.vlm_trust_remote_code),
        )
    )
    rag_llm = _build_local_llm(args)
    model = ImageRAGModel(
        vstore_path=args.vstore,
        captioner=captioner,
        rag_llm=rag_llm,
        model="gpt-4o",
        prompt_log_path=prompt_jsonl_name if args.write else None,
        logprobs_log_path=None,
    )
    print(f"Retriever device: {model.get_device()}")
    rarespp_predictions = await model.rarespecies_dataset_run(
        interval=interval,
        verbose=2,
    )
    overalls, preds = extract_tax_metrics_rs(rarespp_predictions, verbose=True)

    if args.write:
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
        write_preds_to_csv(preds, predictions_csv_name)
        write_per_rank_binary_jsonl(rarespp_predictions, per_rank_jsonl_name)
        overalls_with_hier = dict(overalls)
        for rank, metrics in overalls_with_hier.items():
            if isinstance(metrics, dict) and rank in pr_hm:
                metrics.update(pr_hm[rank])
        overalls_with_hier.update(hm)
        write_overall_metrics(metrics_csv_name, overalls_with_hier)
        write_rank_attempts_csv(
            rank_attempts_csv_name,
            overalls,
            total_samples=len(preds),
        )

        try:
            rsid_to_hier = {}
            rsid_to_binary = {}
            for sample in rarespp_predictions:
                rsid_val = sample.get("RSID")
                if rsid_val is None:
                    continue
                rsid_to_hier[str(rsid_val)] = sample_hierarchical_metrics(
                    sample.get("true_class", {}),
                    sample.get("guess_class", {}),
                )
                try:
                    rsid_to_binary[str(rsid_val)] = sample_binary_accuracy(
                        sample.get("true_class", {}) or {},
                        sample.get("guess_class", {}) or {},
                    )
                except Exception:
                    rsid_to_binary[str(rsid_val)] = None

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
                    if "rsid" not in obj and obj.get("RSID") is not None:
                        obj["rsid"] = obj.get("RSID")
                    obj = {
                        "rsid": obj.get("rsid"),
                        "prompt": obj.get("prompt"),
                        "response": obj.get("response"),
                    }
                    lines_out.append(json.dumps(obj, ensure_ascii=False) + "\n")
            with open(prompt_jsonl_name, "w", encoding="utf-8") as dst:
                for line in lines_out:
                    dst.write(line)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
