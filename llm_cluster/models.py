from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from llm_cluster.keys import get_api_key

Message = Mapping[str, str]


class ChatModel(Protocol):
    """Minimal chat model interface used by the clustering code."""

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> str:
        """Generate one assistant response from role/content chat messages."""


class AsyncChatModel(ChatModel, Protocol):
    async def generate_async(self, messages: Sequence[Message], **kwargs: Any) -> str:
        """Generate one assistant response asynchronously."""


@dataclass
class OpenRouterModel:
    """OpenRouter chat model using the OpenAI Python client."""

    model_name: str
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float | None = 360.0
    api_keys_path: str | os.PathLike[str] | None = None
    default_headers: Mapping[str, str] | None = None
    default_generation_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        resolved_api_key = (
            self.api_key
            or os.getenv("OPENROUTER_API_KEY")
            or get_api_key("openrouter", path=self.api_keys_path)
        )
        if not resolved_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY or api-keys.json is required for "
                "provider='openrouter'."
            )

        from openai import AsyncOpenAI, OpenAI

        headers = dict(self.default_headers or {})
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if site_url:
            headers.setdefault("HTTP-Referer", site_url)
        if app_name:
            headers.setdefault("X-Title", app_name)

        self.default_generation_kwargs.setdefault(
            "extra_body",
            {"reasoning": {"effort": "none", "exclude": True}},
        )

        client_kwargs: dict[str, Any] = {
            "api_key": resolved_api_key,
            "base_url": self.base_url,
        }
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        if headers:
            client_kwargs["default_headers"] = headers

        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> str:
        request_kwargs = {**self.default_generation_kwargs, **kwargs}
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            **request_kwargs,
        )

        return _message_content_to_text(response.choices[0].message.content)

    async def generate_async(self, messages: Sequence[Message], **kwargs: Any) -> str:
        request_kwargs = {**self.default_generation_kwargs, **kwargs}
        response = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            **request_kwargs,
        )

        return _message_content_to_text(response.choices[0].message.content)


@dataclass
class VLLMModel:
    """vLLM OpenAI-compatible chat model."""

    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = 360.0
    default_headers: Mapping[str, str] | None = None
    default_generation_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from openai import AsyncOpenAI, OpenAI

        resolved_api_key = self.api_key or os.getenv("VLLM_API_KEY") or "EMPTY"
        resolved_base_url = _normalize_vllm_base_url(
            self.base_url or os.getenv("VLLM_BASE_URL") or "http://localhost:8100"
        )

        client_kwargs: dict[str, Any] = {
            "api_key": resolved_api_key,
            "base_url": resolved_base_url,
        }
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        if self.default_headers:
            client_kwargs["default_headers"] = dict(self.default_headers)

        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)
        self.model_name = self._resolve_model_name(self.model_name)

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> str:
        request_kwargs = {**self.default_generation_kwargs, **kwargs}
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            **request_kwargs,
        )

        return _message_content_to_text(response.choices[0].message.content)

    async def generate_async(self, messages: Sequence[Message], **kwargs: Any) -> str:
        request_kwargs = {**self.default_generation_kwargs, **kwargs}
        response = await self._async_client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            **request_kwargs,
        )

        return _message_content_to_text(response.choices[0].message.content)

    async def generate_batch_async(
        self,
        messages_batch: Sequence[Sequence[Message]],
        **kwargs: Any,
    ) -> list[str]:
        if not messages_batch:
            return []

        from openai.types.chat import ChatCompletion

        request_kwargs = {**self.default_generation_kwargs, **kwargs}
        response = await self._async_client.post(
            "/chat/completions/batch",
            cast_to=ChatCompletion,
            body={
                "model": self.model_name,
                "messages": [list(messages) for messages in messages_batch],
                **request_kwargs,
            },
        )

        choices = sorted(response.choices, key=lambda choice: choice.index)
        if len(choices) != len(messages_batch):
            raise RuntimeError(
                "vLLM batch response did not include one choice per input."
            )
        return [_message_content_to_text(choice.message.content) for choice in choices]

    def _resolve_model_name(self, model_name: str) -> str:
        normalized = model_name.strip()
        if normalized.lower() not in {"", "auto", "default"}:
            return normalized

        models = self._client.models.list()
        if not models.data:
            raise RuntimeError("vLLM server did not report any models from /v1/models.")
        return models.data[0].id


def load_model(provider: str, model_name: str, **kwargs: Any) -> ChatModel:
    """Create a chat model backend.

    The returned model always exposes `.generate(messages, **kwargs)`.
    """

    normalized_provider = provider.lower().strip()
    if normalized_provider == "openrouter":
        return OpenRouterModel(
            model_name=_normalize_openrouter_model_name(model_name),
            **kwargs,
        )
    if normalized_provider == "vllm":
        return VLLMModel(model_name=model_name, **kwargs)

    raise ValueError(f"Unsupported model provider: {provider!r}")


def _normalize_openrouter_model_name(model_name: str) -> str:
    aliases = {
        "qwen3.5-9b": "qwen/qwen3.5-9b",
    }
    normalized = model_name.strip()
    return aliases.get(normalized.lower(), normalized)


def _normalize_vllm_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    # Some OpenAI-compatible APIs may return content parts instead of a string.
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
