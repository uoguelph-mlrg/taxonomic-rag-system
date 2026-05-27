"""Shared utilities for appending token-level logprobs records to JSONL files."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

DEFAULT_LLM_SEED: int = 12345
DEFAULT_TOP_LOGPROBS: int = 20


def _token_dict(token: Any) -> dict[str, Any]:
    if isinstance(token, dict):
        return token
    return {
        "token": getattr(token, "token", None),
        "logprob": getattr(token, "logprob", None),
        "top_logprobs": getattr(token, "top_logprobs", None),
    }


def _build_token_records(
    tokens_src: list[Any],
) -> tuple[list[dict[str, object]], list[str]]:
    token_records: list[dict[str, object]] = []
    char_cursor = 0
    byte_cursor = 0
    rebuilt: list[str] = []
    for i, raw_tk in enumerate(tokens_src):
        tk = _token_dict(raw_tk)
        tok = tk.get("token")
        lpv = tk.get("logprob")
        top_list = None
        try:
            raw_top = tk.get("top_logprobs")
            if isinstance(raw_top, list):
                top_list = []
                for cand in raw_top:
                    cand_dict = _token_dict(cand)
                    ctok = cand_dict.get("token")
                    clpv = cand_dict.get("logprob")
                    if ctok is not None and clpv is not None:
                        top_list.append({"t": ctok, "lp": clpv})
        except Exception:
            top_list = None
        if tok is None:
            continue
        s_char = char_cursor
        s_byte = byte_cursor
        rebuilt.append(tok)
        char_cursor += len(tok)
        byte_cursor += len(tok.encode("utf-8"))
        token_records.append(
            {
                "idx": i,
                "t": tok,
                "lp": lpv,
                "char_s": s_char,
                "char_e": char_cursor,
                "byte_s": s_byte,
                "byte_e": byte_cursor,
                "top": top_list,
            }
        )
    return token_records, rebuilt


def _find_json_string_value_span(
    raw: str, key: str, start_at: int = 0, end_at: int | None = None
) -> tuple[int, int] | None:
    end_lim = len(raw) if end_at is None else end_at
    kq = f'"{key}"'
    i = raw.find(kq, start_at, end_lim)
    if i == -1:
        return None
    j = raw.find(":", i + len(kq), end_lim)
    if j == -1:
        return None
    j += 1
    while j < end_lim and raw[j] in " \t\r\n":
        j += 1
    if j >= end_lim or raw[j] != '"':
        return None
    val_start = j + 1
    p = val_start
    while p < end_lim:
        c = raw[p]
        if c == "\\":
            p += 2
            continue
        if c == '"':
            return (val_start, p)
        p += 1
    return None


def _find_json_object_span(
    raw: str, key: str, start_at: int = 0
) -> tuple[int, int] | None:
    kq = f'"{key}"'
    i = raw.find(kq, start_at)
    if i == -1:
        return None
    j = raw.find(":", i + len(kq))
    if j == -1:
        return None
    j += 1
    while j < len(raw) and raw[j] in " \t\r\n":
        j += 1
    if j >= len(raw) or raw[j] != "{":
        return None
    depth = 0
    in_str = False
    p = j
    while p < len(raw):
        ch = raw[p]
        if in_str:
            if ch == "\\":
                p += 2
                continue
            if ch == '"':
                in_str = False
            p += 1
            continue
        if ch == '"':
            in_str = True
            p += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (j, p + 1)
        p += 1
    return None


def _token_range_for_char_range(
    recs: list[dict[str, object]], cr: tuple[int, int] | None
) -> tuple[int, int] | None:
    if not cr:
        return None
    s_char, e_char = cr
    idxs: list[int] = []
    for r in recs:
        rs = int(r["char_s"])  # type: ignore[arg-type]
        re = int(r["char_e"])  # type: ignore[arg-type]
        if re > s_char and rs < e_char:
            idxs.append(int(r["idx"]))  # type: ignore[arg-type]
    if not idxs:
        return None
    return (min(idxs), max(idxs) + 1)


def _build_sections(
    full_text: str, token_records: list[dict[str, object]]
) -> dict[str, object]:
    sections: dict[str, object] = {}
    for key in ("ancestral", "specific", "commentary", "bio_knowledge"):
        cr = _find_json_string_value_span(full_text, key)
        tr = _token_range_for_char_range(token_records, cr) if token_records else None
        if cr is not None:
            sections[key] = {
                "char_range": list(cr),
                "token_range": list(tr) if tr else None,
            }

    class_obj_span = _find_json_object_span(full_text, "classification")
    class_map: dict[str, object] = {}
    if class_obj_span is not None:
        c_s, c_e = class_obj_span
        for rk in ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"):
            scr = _find_json_string_value_span(full_text, rk, start_at=c_s, end_at=c_e)
            tr = (
                _token_range_for_char_range(token_records, scr)
                if token_records
                else None
            )
            if scr is not None:
                class_map[rk] = {
                    "char_range": list(scr),
                    "token_range": list(tr) if tr else None,
                }
        sections["classification"] = class_map
    return sections


def append_logprobs_jsonl(
    *,
    target_path: str,
    prompt: str,
    response_text: str,
    tokens_src: list[Any],
    rsid: str | None = None,
    model_name: str = "",
    created: int | None = None,
    finish_reason: str | None = None,
    system_fingerprint: str | None = None,
    schema_version: str = "logprob_v1",
    gen_params: dict[str, Any] | None = None,
    prompt_id: str | None = None,
    sample_idx: int | None = None,
    sample_seed: int | None = None,
) -> None:
    """Append a token-level logprobs record to the configured JSONL file."""
    if not target_path:
        return

    try:
        Path(target_path).expanduser().parent.mkdir(parents=True, exist_ok=True)

        token_records, rebuilt = _build_token_records(tokens_src)
        full_text = response_text or ""
        rebuilt_text = "".join(rebuilt)
        warnings: list[str] = []
        if full_text and rebuilt_text and rebuilt_text != full_text:
            warnings.append("rebuilt_text_mismatch")

        total_tokens = len(token_records)
        missing_token_count = 0
        for r in token_records:
            lpv_r = r.get("lp")
            if lpv_r is None or lpv_r == -9999.0:
                missing_token_count += 1
        missing_token_ratio = (
            (missing_token_count / total_tokens) if total_tokens else 0.0
        )

        sections = _build_sections(full_text, token_records)
        resp_hash = "sha256:" + hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        record = {
            "schema_version": schema_version,
            "rsid": rsid,
            "prompt_id": prompt_id,
            "sample_idx": sample_idx,
            "sample_seed": sample_seed,
            "model": model_name,
            "created": created,
            "finish_reason": finish_reason,
            "system_fingerprint": system_fingerprint,
            "response_text": full_text,
            "response_text_hash": resp_hash,
            "gen_params": gen_params
            or {
                "logprobs": True,
                "temperature": 1,
                "top_p": 1,
                "seed": DEFAULT_LLM_SEED,
                "top_logprobs": DEFAULT_TOP_LOGPROBS,
            },
            "tokens": token_records,
            "sections": sections,
            "quality": {
                "missing_token_count (lp=-9999.0)": missing_token_count,
                "total_token_count": total_tokens,
                "missing_token_ratio": missing_token_ratio,
            },
            "warnings": warnings,
        }

        with open(target_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as log_err:
        logger.warning(f"Failed to log logprobs: {log_err}")


def _extract_tokens_from_langchain_metadata(meta: dict[str, Any]) -> list[Any]:
    tokens_src: list[Any] = []
    try:
        lp = meta.get("logprobs")
        if isinstance(lp, dict) and isinstance(lp.get("content"), list):
            tokens_src = lp["content"]
        elif isinstance(meta.get("choices"), list):
            ch0 = meta["choices"][0] if meta["choices"] else {}
            lp2 = ch0.get("logprobs") if isinstance(ch0, dict) else None
            if isinstance(lp2, dict) and isinstance(lp2.get("content"), list):
                tokens_src = lp2["content"]
    except Exception:
        tokens_src = []
    return tokens_src


def append_logprobs_from_langchain_message(
    *,
    target_path: str,
    prompt: str,
    ai_message: Any,
    rsid: str | None = None,
    path_override: str | None = None,
    prompt_id: str | None = None,
    sample_idx: int | None = None,
    sample_seed: int | None = None,
    schema_version: str = "logprob_v1",
    gen_params_override: dict[str, Any] | None = None,
) -> None:
    """Append logprobs extracted from a LangChain AIMessage."""
    resolved_path = path_override or target_path
    if not resolved_path:
        return

    full_text = getattr(ai_message, "content", "") or ""
    meta = getattr(ai_message, "response_metadata", {}) or {}
    model_name = meta.get("model_name") or meta.get("model") or ""
    created = meta.get("created") or None
    finish_reason = meta.get("finish_reason") or (
        meta.get("choices", [{}])[0].get("finish_reason")
        if isinstance(meta.get("choices"), list) and meta.get("choices")
        else None
    )
    system_fingerprint = meta.get("system_fingerprint") or None
    tokens_src = _extract_tokens_from_langchain_metadata(meta)

    append_logprobs_jsonl(
        target_path=resolved_path,
        prompt=prompt,
        response_text=full_text,
        tokens_src=tokens_src,
        rsid=rsid,
        model_name=model_name,
        created=created,
        finish_reason=finish_reason,
        system_fingerprint=system_fingerprint,
        schema_version=schema_version,
        gen_params=gen_params_override,
        prompt_id=prompt_id,
        sample_idx=sample_idx,
        sample_seed=sample_seed,
    )


def append_logprobs_from_openai_completion(
    *,
    target_path: str,
    prompt: str,
    completion: Any,
    rsid: str | None = None,
    schema_version: str = "logprob_v1",
    gen_params_override: dict[str, Any] | None = None,
) -> None:
    """Append logprobs extracted from an OpenAI ChatCompletion response."""
    if not target_path:
        return

    choices = getattr(completion, "choices", None) or []
    choice0 = choices[0] if choices else None
    message = getattr(choice0, "message", None) if choice0 is not None else None
    full_text = getattr(message, "content", "") or ""

    logprobs_obj = getattr(choice0, "logprobs", None) if choice0 is not None else None
    tokens_src: list[Any] = []
    if logprobs_obj is not None:
        content = getattr(logprobs_obj, "content", None)
        if isinstance(content, list):
            tokens_src = content

    append_logprobs_jsonl(
        target_path=target_path,
        prompt=prompt,
        response_text=full_text,
        tokens_src=tokens_src,
        rsid=rsid,
        model_name=getattr(completion, "model", "") or "",
        created=getattr(completion, "created", None),
        finish_reason=getattr(choice0, "finish_reason", None) if choice0 else None,
        system_fingerprint=getattr(completion, "system_fingerprint", None),
        schema_version=schema_version,
        gen_params=gen_params_override,
    )
