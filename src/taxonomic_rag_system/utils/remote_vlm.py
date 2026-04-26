"""Remote OpenAI-compatible VLM utilities for descriptive image captioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from taxonomic_rag_system.utils.local_vlm import (
    DESCRIPTIVE_CAPTION_SYSTEM_PROMPT,
    DESCRIPTIVE_CAPTION_USER_PROMPT,
)


@dataclass(frozen=True)
class RemoteVLMConfig:
    """Configuration for an OpenAI-compatible remote VLM endpoint."""

    model: str
    base_url: str
    api_key: str = "EMPTY"
    max_tokens: int = 1024
    temperature: float | None = None
    timeout: float | None = 120.0


class RemoteVLMCaptioner:
    """Generate descriptive captions using an OpenAI-compatible VLM server."""

    def __init__(self, cfg: RemoteVLMConfig) -> None:
        self.cfg = cfg
        self._client: Any | None = None

    def _client_instance(self) -> Any:
        """Create the async OpenAI-compatible client lazily."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                timeout=self.cfg.timeout,
            )
        return self._client

    @staticmethod
    def _image_data_url(image_b64: str) -> str:
        """Convert a raw base64 image into an OpenAI image_url data URL."""
        if image_b64.startswith("data:image/"):
            return image_b64
        return f"data:image/jpeg;base64,{image_b64}"

    @staticmethod
    def _message_content_text(content: Any) -> str:
        """Extract text content from OpenAI-compatible message payloads."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    async def generate_caption(self, image_b64: str) -> str:
        """Generate a detailed caption for a base64-encoded image."""
        try:
            kwargs: dict[str, Any] = {
                "model": self.cfg.model,
                "messages": [
                    {
                        "role": "system",
                        "content": DESCRIPTIVE_CAPTION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": DESCRIPTIVE_CAPTION_USER_PROMPT,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self._image_data_url(image_b64),
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": self.cfg.max_tokens,
            }
            if self.cfg.temperature is not None:
                kwargs["temperature"] = self.cfg.temperature

            response = await self._client_instance().chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return self._message_content_text(content).strip()
        except Exception as e:
            print(f"Error during remote caption generation: {e}")
            return ""
