"""
Client for Ollama's native API.

Ollama returns thinking and answer as separate fields, which is
exactly what we want for Chain-of-Thought workflows.
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from regulaai.config import get_settings
from regulaai.thinking import ThinkingResponse


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        s = get_settings()
        # Strip /v1 if present — we use the native API, not OpenAI-compat
        url = (base_url or s.ollama_base_url).rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self._base_url = url
        self._model = model or s.ollama_model
        self._settings = s

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def chat(
        self,
        prompt: str,
        system: str | None = None,
        enable_thinking: bool | None = None,
    ) -> ThinkingResponse:
        s = self._settings
        think = s.enable_thinking if enable_thinking is None else enable_thinking

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {
                "temperature": s.temperature,
                "top_p": s.top_p,
                "num_predict": s.max_output_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("message", {})
        return ThinkingResponse(
            thinking=msg.get("thinking", "") or "",
            answer=msg.get("content", "") or "",
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            completion_tokens=data.get("eval_count", 0) or 0,
        )
