"""Shared Ollama client.

Three modules used to carry their own copy of "POST to /api/chat, strip the
markdown fence, json.loads it".  They now share this one, which also keeps a
single pooled ``httpx.AsyncClient`` instead of building a new connection pool
for every inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class OllamaUnavailable(RuntimeError):
    """Raised when Ollama cannot be reached or the model is missing."""


def extract_json(text: str) -> str:
    """Pull a JSON document out of a possibly fenced / chatty LLM response."""
    text = (text or "").strip()

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    # Fall back to the outermost brace pair, which survives models that prefix
    # the answer with prose.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]

    return text.strip()


class OllamaClient:
    """Minimal async Ollama wrapper with a shared connection pool."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "gemma3",
        timeout: float = 120.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=self.timeout)
            return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def available(self, timeout: float = 5.0) -> tuple[bool, list[str]]:
        """Return (model_present, available_model_names)."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.host}/api/tags", timeout=timeout)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except Exception as exc:
            logger.debug("Ollama availability check failed: %s", exc)
            return False, []
        present = any(n == self.model or n.startswith(f"{self.model}:") for n in names)
        return present, names

    async def chat(
        self, system: str, user: str, num_predict: int = 1000, temperature: float = 0.1
    ) -> str:
        """Run a single-turn chat completion and return the assistant text."""
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": num_predict},
                },
            )
            response.raise_for_status()
        except Exception as exc:
            raise OllamaUnavailable(str(exc)) from exc

        data = response.json()
        # Thinking models put reasoning in 'thinking' and the answer in 'content'.
        content = data.get("message", {}).get("content", "")
        return content or data.get("response", "")

    async def chat_json(self, system: str, user: str, **kwargs) -> dict[str, Any]:
        """Chat and parse the response as JSON."""
        raw = await self.chat(system, user, **kwargs)
        try:
            return json.loads(extract_json(raw))
        except json.JSONDecodeError as exc:
            raise OllamaUnavailable(f"model returned unparseable JSON: {raw[:200]!r}") from exc

    async def vision_json(
        self, prompt: str, image_b64: str, num_predict: int = 300
    ) -> dict[str, Any]:
        """Run a vision prompt against an image and parse the JSON response."""
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": num_predict},
                },
            )
            response.raise_for_status()
        except Exception as exc:
            raise OllamaUnavailable(str(exc)) from exc

        raw = response.json().get("response", "")
        try:
            return json.loads(extract_json(raw))
        except json.JSONDecodeError as exc:
            raise OllamaUnavailable(f"model returned unparseable JSON: {raw[:200]!r}") from exc
