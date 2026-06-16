"""
Local (in-process) LLM utilities for running open-source models on HPC.

This module is intentionally implemented with lazy imports so the base project
can run without the optional local-LLM dependencies installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LocalLLMConfig:
    """Configuration for building a local Hugging Face text-generation LLM."""

    model_id: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    # "auto" / HF literal / alias (e.g. cuda0); dict is forwarded to ``from_pretrained``
    device_map: str | dict[str, Any] = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True

    # When True, generated text is completion only (not prompt+completion).
    return_full_text: bool = False


def _apply_generation_defaults(
    pipeline_kwargs: dict[str, Any],
    cfg: LocalLLMConfig,
) -> dict[str, Any]:
    """Populate HF generation kwargs while allowing model defaults to pass through."""
    pk = dict(pipeline_kwargs)

    pk.setdefault("return_full_text", cfg.return_full_text)
    if cfg.max_new_tokens is not None:
        pk.setdefault("max_new_tokens", cfg.max_new_tokens)

    # `None` means "leave this entirely to the model's generation_config".
    if cfg.temperature is None:
        return pk

    # HF text-generation pipelines reject temperature=0.0 even for greedy decoding.
    # Only pass sampling-specific knobs when sampling is actually enabled.
    do_sample = cfg.temperature > 0
    pk.setdefault("do_sample", do_sample)
    if do_sample:
        pk.setdefault("temperature", cfg.temperature)
        if cfg.top_p is not None:
            pk.setdefault("top_p", cfg.top_p)

    return pk


def normalize_device_map(device_map: str | dict[str, Any]) -> str | dict[str, Any]:
    """Resolve ``device_map`` for ``AutoModelForCausalLM.from_pretrained``.

    ``device_map="auto"`` can place some layers on CPU under memory pressure, which
    then breaks forward passes (CUDA/CPU tensor mismatch). For a single GPU, prefer
    ``{"": 0}`` so the full weights stay on GPU 0.

    Parameters
    ----------
    device_map
        Hugging Face ``device_map`` string (e.g. ``"auto"``), a JSON object string
        parsed to a dict, or a dict. Aliases ``cuda0``, ``gpu0``, ``single``, ``0``,
        and ``cuda:0`` map to ``{"": 0}``.

    Returns
    -------
    str | dict[str, Any]
        Value suitable for ``from_pretrained(..., device_map=...)``.
    """
    if isinstance(device_map, dict):
        return device_map
    s = str(device_map).strip()
    if not s:
        return "auto"
    key = s.lower()
    single_aliases = frozenset({"cuda0", "gpu0", "single", "cuda:0", "0"})
    if key in single_aliases:
        return {"": 0}
    if s.startswith("{"):
        try:
            parsed: Any = json.loads(s)
        except json.JSONDecodeError as e:
            msg = f"device_map must be valid JSON when starting with '{{': {e}"
            raise ValueError(msg) from e
        if not isinstance(parsed, dict):
            raise ValueError("device_map JSON must decode to an object")
        return parsed
    return s


def _load_causal_lm(model_id: str, mk: dict[str, Any]) -> Any:
    """Load a causal LM, with a Qwen3.5-family fallback when AutoModel fails."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **mk)
    except (ValueError, OSError, TypeError) as exc:
        try:
            from transformers import Qwen3_5ForCausalLM

            return Qwen3_5ForCausalLM.from_pretrained(model_id, **mk)
        except Exception:
            raise exc


def _apply_thinking_disable(model: Any, pipeline_kwargs: dict[str, Any]) -> None:
    """Disable Qwen thinking mode when requested via pipeline kwargs."""
    if pipeline_kwargs.get("enable_thinking") is not False:
        return
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and hasattr(gen_cfg, "enable_thinking"):
        gen_cfg.enable_thinking = False
    pipeline_kwargs.pop("enable_thinking", None)


def build_hf_textgen_llm(
    *,
    cfg: LocalLLMConfig,
    model_kwargs: Optional[dict[str, Any]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """Build a LangChain-compatible LLM backed by HF `pipeline(text-generation)`.

    Parameters
    ----------
    cfg
        Model + generation configuration.
    model_kwargs
        Extra keyword arguments forwarded to `from_pretrained` (e.g., quantization).
    pipeline_kwargs
        Extra keyword arguments forwarded to `transformers.pipeline`.

    Returns
    -------
    Any
        A LangChain LLM runnable (typically `HuggingFacePipeline`).

    Notes
    -----
    - Imports are lazy to keep transformers as an optional dependency.
    - This uses a plain text-generation pipeline. The caller is responsible for
      providing a prompt that elicits strict JSON for downstream parsing.
    """
    # Lazy imports (optional dependency)
    import importlib

    from transformers import (
        AutoTokenizer,
        pipeline,
    )

    try:
        _lc_llms = importlib.import_module("langchain_community.llms")
    except Exception:  # pragma: no cover
        _lc_llms = importlib.import_module("langchain.llms")
    hf_pipeline_cls = _lc_llms.HuggingFacePipeline

    mk: dict[str, Any] = dict(model_kwargs or {})
    pk: dict[str, Any] = dict(pipeline_kwargs or {})

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        use_fast=True,
        trust_remote_code=cfg.trust_remote_code,
    )

    effective_dm = mk.pop("device_map", cfg.device_map)
    resolved_dm = normalize_device_map(effective_dm)
    mk["device_map"] = resolved_dm
    mk.setdefault("torch_dtype", cfg.torch_dtype)
    mk.setdefault("trust_remote_code", cfg.trust_remote_code)
    model = _load_causal_lm(cfg.model_id, mk)
    _apply_thinking_disable(model, pk)

    pk = _apply_generation_defaults(pk, cfg)
    # Put pipeline inputs on GPU 0 when the full model uses {"": 0}.
    if resolved_dm == {"": 0}:
        pk.setdefault("device", 0)

    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **pk,
    )

    return hf_pipeline_cls(pipeline=gen_pipe)
