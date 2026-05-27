"""Unit tests for shared logprobs JSONL logging utilities."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from taxonomic_rag_system.utils.logprobs_logging import (
    append_logprobs_from_openai_completion,
    append_logprobs_jsonl,
)


def test_append_logprobs_jsonl_writes_expected_record(tmp_path: Path) -> None:
    target = tmp_path / "logprobs.jsonl"
    response_text = json.dumps(
        {
            "classification": {
                "Kingdom": "Animalia",
                "Phylum": "Arthropoda",
                "Class": "Insecta",
                "Order": "N/A",
                "Family": "N/A",
                "Genus": "N/A",
                "Species": "N/A",
            }
        }
    )
    tokens_src = [
        {"token": "{", "logprob": -0.1, "top_logprobs": []},
        {"token": "Kingdom", "logprob": -0.2, "top_logprobs": []},
    ]

    append_logprobs_jsonl(
        target_path=str(target),
        prompt="system + user prompt",
        response_text=response_text,
        tokens_src=tokens_src,
        rsid="RS123",
        model_name="gpt-4o",
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["schema_version"] == "logprob_v1"
    assert record["rsid"] == "RS123"
    assert record["model"] == "gpt-4o"
    assert record["tokens"]
    assert "classification" in record["sections"]
    assert "Kingdom" in record["sections"]["classification"]


def test_append_logprobs_jsonl_noop_when_target_empty() -> None:
    append_logprobs_jsonl(
        target_path="",
        prompt="prompt",
        response_text="{}",
        tokens_src=[],
        rsid="RS123",
    )


def test_append_logprobs_from_openai_completion(tmp_path: Path) -> None:
    target = tmp_path / "openai_logprobs.jsonl"
    response_text = json.dumps(
        {
            "classification": {
                "Kingdom": "Animalia",
                "Phylum": "Chordata",
                "Class": "Mammalia",
                "Order": "N/A",
                "Family": "N/A",
                "Genus": "N/A",
                "Species": "N/A",
            }
        }
    )

    mock_message = MagicMock()
    mock_message.content = response_text

    mock_logprob_token = MagicMock()
    mock_logprob_token.token = "{"
    mock_logprob_token.logprob = -0.05
    mock_logprob_token.top_logprobs = []

    mock_logprobs = MagicMock()
    mock_logprobs.content = [mock_logprob_token]

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"
    mock_choice.logprobs = mock_logprobs

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.model = "gpt-4o"
    mock_completion.created = 1234567890
    mock_completion.system_fingerprint = "fp-test"

    append_logprobs_from_openai_completion(
        target_path=str(target),
        prompt="prompt text",
        completion=mock_completion,
        rsid="RS456",
    )

    record = json.loads(target.read_text(encoding="utf-8").strip())
    assert record["rsid"] == "RS456"
    assert record["model"] == "gpt-4o"
    assert record["response_text"] == response_text
    assert record["tokens"]
