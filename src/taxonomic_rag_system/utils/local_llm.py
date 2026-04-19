"""
Local (in-process) LLM utilities for running open-source models on HPC.

This module is intentionally implemented with lazy imports so the base project
can run without the optional local-LLM dependencies installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LocalLLMConfig:
    """Configuration for building a local Hugging Face text-generation LLM."""

    model_id: str
    max_new_tokens: int = 900
    temperature: float = 0.0
    top_p: float = 1.0
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True

    # When True, the generated text contains ONLY the completion (not prompt+completion).
    return_full_text: bool = False


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
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )

    try:
        from langchain_community.llms import HuggingFacePipeline  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        # Older/newer LangChain distributions may locate this elsewhere.
        from langchain.llms import HuggingFacePipeline  # type: ignore[import-not-found]

    mk: dict[str, Any] = dict(model_kwargs or {})
    pk: dict[str, Any] = dict(pipeline_kwargs or {})

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

    mk.setdefault("device_map", cfg.device_map)
    mk.setdefault("torch_dtype", cfg.torch_dtype)
    mk.setdefault("trust_remote_code", cfg.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **mk)

    do_sample = cfg.temperature > 0

    pk.setdefault("max_new_tokens", cfg.max_new_tokens)
    pk.setdefault("do_sample", do_sample)
    pk.setdefault("temperature", cfg.temperature)
    pk.setdefault("top_p", cfg.top_p)
    pk.setdefault("return_full_text", cfg.return_full_text)

    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **pk,
    )

    return HuggingFacePipeline(pipeline=gen_pipe)

