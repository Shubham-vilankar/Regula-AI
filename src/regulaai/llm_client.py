"""
Client for Ollama's native API with Langfuse v4 (OpenTelemetry) tracing.

Every call is logged to Langfuse with thinking, answer, and token usage.
"""

from __future__ import annotations

import httpx
from langfuse import Langfuse, get_client
from tenacity import retry, stop_after_attempt, wait_exponential

from regulaai.config import get_settings
from regulaai.thinking import ThinkingResponse


_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        s = get_settings()
        _langfuse = Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            base_url=s.langfuse_base_url,
        )
    return _langfuse


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        s = get_settings()
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
        trace_name: str = "ollama-chat",
    ) -> ThinkingResponse:
        s = self._settings
        think = s.enable_thinking if enable_thinking is None else enable_thinking

        langfuse = get_langfuse()

        with langfuse.start_as_current_observation(
            as_type="span", name=trace_name
        ) as span:
            span.update(input={"system": system, "prompt": prompt})

            with span.start_as_current_observation(
                as_type="generation",
                name="chat",
                model=self._model,
                input={"system": system, "prompt": prompt},
            ) as generation:
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
                result = ThinkingResponse(
                    thinking=msg.get("thinking", "") or "",
                    answer=msg.get("content", "") or "",
                    prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                    completion_tokens=data.get("eval_count", 0) or 0,
                    trace_id=span.trace_id,
                )

                generation.update(
                    output={"thinking": result.thinking, "answer": result.answer},
                    usage_details={
                        "input": result.prompt_tokens,
                        "output": result.completion_tokens,
                    },
                )
                span.update(
                    output={"thinking": result.thinking, "answer": result.answer}
                )

        langfuse.flush()
        return result
