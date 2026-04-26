"""Tests for remote OpenAI-compatible VLM captioning."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from taxonomic_rag_system.utils.remote_vlm import RemoteVLMCaptioner, RemoteVLMConfig


@pytest.mark.asyncio
async def test_remote_vlm_captioner_sends_openai_multimodal_request(monkeypatch):
    """The remote captioner should call chat completions with a data URL image."""
    captured: dict[str, object] = {}

    class _FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=" remote caption "),
                    ),
                ],
            )

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)

    captioner = RemoteVLMCaptioner(
        RemoteVLMConfig(
            model="Qwen2.5-VL-7B-Instruct",
            base_url="http://gpu001:8000/v1",
            api_key="EMPTY",
            max_tokens=128,
        ),
    )

    caption = await captioner.generate_caption("abc123")

    assert caption == "remote caption"
    assert captured["model"] == "Qwen2.5-VL-7B-Instruct"
    assert captured["max_tokens"] == 128
    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    image_part = messages[1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"] == "data:image/jpeg;base64,abc123"
    assert captured["client_kwargs"]["base_url"] == "http://gpu001:8000/v1"
    assert captured["client_kwargs"]["api_key"] == "EMPTY"


@pytest.mark.asyncio
async def test_caption_runner_writes_jsonl_with_rsid_and_caption(tmp_path, monkeypatch):
    """The caption-only runner should write one RAG-ready JSONL record per sample."""
    from taxonomic_rag_system.runs import rare_species_caption_vecinf_vlm as runner

    class _FakeCaptioner:
        def __init__(self, cfg):
            self.cfg = cfg

        async def generate_caption(self, image_b64: str) -> str:
            assert image_b64 == "image-b64"
            return "Detailed remote caption"

    monkeypatch.setattr(
        runner,
        "_build_dataloader",
        lambda **_: [
            (
                [object()],
                [
                    {
                        "RSID": "RS001",
                        "Kingdom": "Animalia",
                        "Phylum": "Arthropoda",
                        "Class": "Insecta",
                    },
                ],
            ),
        ],
    )
    monkeypatch.setattr(runner.ImageProcessor, "process_image", lambda **_: "image-b64")
    monkeypatch.setattr(runner, "RemoteVLMCaptioner", _FakeCaptioner)

    output_jsonl = tmp_path / "captions.jsonl"
    metadata_json = tmp_path / "metadata.json"
    args = argparse.Namespace(
        server_job_id="",
        base_url="http://gpu001:8000/v1",
        model="Qwen2.5-VL-7B-Instruct",
        output_jsonl=str(output_jsonl),
        metadata_json=str(metadata_json),
        interval_start=0,
        interval_end=1,
        batch_size=1,
        max_tokens=1024,
        temperature=None,
        api_key="EMPTY",
        resume=False,
    )

    metadata = await runner.run_captioning(args)

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert rows == [
        {
            "rsid": "RS001",
            "caption": "Detailed remote caption",
            "true_class": {
                "Kingdom": "Animalia",
                "Phylum": "Arthropoda",
                "Class": "Insecta",
            },
            "model": "Qwen2.5-VL-7B-Instruct",
            "base_url": "http://gpu001:8000/v1",
            "status": "ok",
            "error": None,
            "interval_start": 0,
            "interval_end": 1,
            "created_at": rows[0]["created_at"],
        },
    ]
    assert metadata["records_total"] == 1
    assert metadata["records_ok"] == 1
    assert metadata["records_error"] == 0
    written_metadata = json.loads(metadata_json.read_text())
    assert written_metadata["output_jsonl"] == str(output_jsonl)
