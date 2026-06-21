from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from llm_cluster.models import ChatModel, Message

COMPARISON_PROMPT_VERSION = "distance-comparison-v1"
DEFAULT_COMPARISON_CACHE_PATH = ".cache/llm-cluster/comparisons.jsonl"
ComparisonInput = tuple[str, str, str, str]

COMPARISON_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "distance_comparison",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reasoning": {"type": "string", "maxLength": 200},
                "is_ab_less_than_cd": {"type": "boolean"},
            },
            "required": ["reasoning", "is_ab_less_than_cd"],
        },
    },
}


@dataclass(frozen=True)
class ComparisonResult:
    reasoning: str
    is_ab_less_than_cd: bool


@dataclass
class ComparisonCache:
    path: str | Path = DEFAULT_COMPARISON_CACHE_PATH

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._records: dict[str, ComparisonResult] = {}
        self._lock = RLock()
        self._load()

    def get(self, key: str) -> ComparisonResult | None:
        with self._lock:
            return self._records.get(key)

    def set(self, key: str, result: ComparisonResult) -> ComparisonResult:
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return existing

            self._records[key] = result
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "key": key,
                            "reasoning": result.reasoning,
                            "is_ab_less_than_cd": result.is_ab_less_than_cd,
                        },
                        ensure_ascii=False,
                    )
                )
                file.write("\n")
            return result

    def _load(self) -> None:
        if not self.path.exists():
            return

        with self.path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record["key"]
                self._records[key] = ComparisonResult(
                    reasoning=record["reasoning"],
                    is_ab_less_than_cd=record["is_ab_less_than_cd"],
                )


@dataclass
class LLMDistanceComparator:
    """LLM-backed comparator for deciding whether d(a, b) < d(c, d)."""

    model: ChatModel
    temperature: float = 0.0
    max_tokens: int = 1024
    max_concurrency: int = 4
    parse_retries: int = 2
    cache: ComparisonCache | None = field(default_factory=ComparisonCache)

    def compare(self, a: str, b: str, c: str, d: str) -> ComparisonResult:
        cache = self.cache
        cache_key = comparison_cache_key(a, b, c, d)
        if cache is not None:
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        result = self._generate_and_parse(a, b, c, d)
        if cache is not None:
            result = cache.set(cache_key, result)
        return result

    def compare_batch(
        self,
        comparisons: list[ComparisonInput],
        *,
        max_concurrency: int | None = None,
    ) -> list[ComparisonResult]:
        concurrency = max_concurrency if max_concurrency is not None else self.max_concurrency
        if concurrency > 1 and hasattr(self.model, "generate_async"):
            return asyncio.run(
                self.compare_batch_async(
                    comparisons,
                    max_concurrency=concurrency,
                )
            )
        return self._compare_batch_sequential(comparisons)

    async def compare_batch_async(
        self,
        comparisons: list[ComparisonInput],
        *,
        max_concurrency: int | None = None,
    ) -> list[ComparisonResult]:
        if not comparisons:
            return []

        results: list[ComparisonResult | None] = [None] * len(comparisons)
        pending_by_key: dict[str, tuple[ComparisonInput, list[int]]] = {}

        for index, comparison in enumerate(comparisons):
            cache_key = comparison_cache_key(*comparison)
            cached_result = self.cache.get(cache_key) if self.cache is not None else None
            if cached_result is not None:
                results[index] = cached_result
                continue

            pending = pending_by_key.get(cache_key)
            if pending is None:
                pending_by_key[cache_key] = (comparison, [index])
            else:
                pending[1].append(index)

        if pending_by_key:
            concurrency = max_concurrency if max_concurrency is not None else self.max_concurrency
            concurrency = min(max(1, concurrency), len(pending_by_key))
            if concurrency == 1 or not hasattr(self.model, "generate_async"):
                for comparison, indexes in pending_by_key.values():
                    result = self.compare(*comparison)
                    for index in indexes:
                        results[index] = result
            else:
                semaphore = asyncio.Semaphore(concurrency)
                tasks = [
                    (
                        asyncio.create_task(
                            self.compare_async(*comparison, semaphore=semaphore)
                        ),
                        indexes,
                    )
                    for comparison, indexes in pending_by_key.values()
                ]
                for task, indexes in tasks:
                    result = await task
                    for index in indexes:
                        results[index] = result

        resolved_results: list[ComparisonResult] = []
        for result in results:
            if result is None:
                raise RuntimeError("Comparison batch did not produce every result.")
            resolved_results.append(result)
        return resolved_results

    async def compare_async(
        self,
        a: str,
        b: str,
        c: str,
        d: str,
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> ComparisonResult:
        cache = self.cache
        cache_key = comparison_cache_key(a, b, c, d)
        if cache is not None:
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        if not hasattr(self.model, "generate_async"):
            return self.compare(a, b, c, d)

        result = await self._generate_and_parse_async(a, b, c, d, semaphore=semaphore)
        if cache is not None:
            result = cache.set(cache_key, result)
        return result

    def _compare_batch_sequential(
        self,
        comparisons: list[ComparisonInput],
    ) -> list[ComparisonResult]:
        return [self.compare(*comparison) for comparison in comparisons]

    def _generate_and_parse(self, a: str, b: str, c: str, d: str) -> ComparisonResult:
        messages = build_distance_comparison_messages(a, b, c, d)
        last_error: Exception | None = None
        for _ in range(self.parse_retries + 1):
            raw_response = self.model.generate(
                messages,
                response_format=COMPARISON_RESPONSE_FORMAT,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            try:
                return parse_comparison_response(raw_response)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        if last_error is None:
            raise RuntimeError("Comparison generation failed without an error.")
        raise last_error

    async def _generate_and_parse_async(
        self,
        a: str,
        b: str,
        c: str,
        d: str,
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> ComparisonResult:
        messages = build_distance_comparison_messages(a, b, c, d)
        last_error: Exception | None = None
        for _ in range(self.parse_retries + 1):
            if semaphore is None:
                raw_response = await self.model.generate_async(
                    messages,
                    response_format=COMPARISON_RESPONSE_FORMAT,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                async with semaphore:
                    raw_response = await self.model.generate_async(
                        messages,
                        response_format=COMPARISON_RESPONSE_FORMAT,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
            try:
                return parse_comparison_response(raw_response)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        if last_error is None:
            raise RuntimeError("Comparison generation failed without an error.")
        raise last_error


def compare(
    a: str,
    b: str,
    c: str,
    d: str,
    *,
    comparator: LLMDistanceComparator,
) -> bool:
    """Return whether d(a, b) is less than d(c, d)."""

    return comparator.compare(a, b, c, d).is_ab_less_than_cd


def build_distance_comparison_messages(
    a: str,
    b: str,
    c: str,
    d: str,
) -> list[Message]:
    user_content = f"""Compare semantic distance between two pairs of user queries.

Distance should mean how different the queries are in intent and requested task.
Lower distance means the two queries are more likely to belong to the same intent category.

Pair AB:
A: {a}
B: {b}

Pair CD:
C: {c}
D: {d}

Question: is d(A, B) strictly less than d(C, D)?"""

    return [
        {
            "role": "system",
            "content": (
                "You are a careful semantic-distance judge. Return only JSON in "
                'this exact shape: {"reasoning": "<one short sentence under 200 chars>", '
                '"is_ab_less_than_cd": <boolean>}. The boolean is true when '
                "d(A, B) is strictly less than d(C, D), and false otherwise. "
                "Do not include any other fields, markdown, or text."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def comparison_cache_key(a: str, b: str, c: str, d: str) -> str:
    return json.dumps(
        {
            "version": COMPARISON_PROMPT_VERSION,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def parse_comparison_response(raw_response: str) -> ComparisonResult:
    payload = _load_json_object(raw_response)
    reasoning = payload.get("reasoning")
    is_ab_less_than_cd = payload.get("is_ab_less_than_cd")

    if not isinstance(reasoning, str):
        raise ValueError("Comparison response field 'reasoning' must be a string.")
    if not isinstance(is_ab_less_than_cd, bool):
        raise ValueError(
            "Comparison response field 'is_ab_less_than_cd' must be a boolean."
        )

    return ComparisonResult(
        reasoning=reasoning,
        is_ab_less_than_cd=is_ab_less_than_cd,
    )


def _load_json_object(raw_response: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        start = raw_response.find("{")
        if start < 0:
            raise ValueError("Comparison response did not contain a JSON object.")
        decoder = json.JSONDecoder()
        payload, _ = decoder.raw_decode(raw_response[start:])

    if not isinstance(payload, dict):
        raise ValueError("Comparison response must be a JSON object.")
    return payload
