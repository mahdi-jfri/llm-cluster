from __future__ import annotations

import asyncio
import atexit
import json
import sqlite3
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from llm_cluster.models import ChatModel, Message

COMPARISON_PROMPT_VERSION = "distance-comparison-v1"
DEFAULT_COMPARISON_CACHE_PATH = ".cache/llm-cluster/comparisons.sqlite"
ComparisonInput = tuple[str, str, str, str]
ComparisonProgressCallback = Callable[[str], None]

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
    sync_interval_seconds: float = 5.0
    flush_batch_size: int = 1000

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()
        self._records = self._load_records()
        self._pending_records: dict[str, ComparisonResult] = {}
        self._last_sync_at = time.monotonic()
        self._closed = False
        atexit.register(self.close)

    def get(self, key: str) -> ComparisonResult | None:
        with self._lock:
            return self._records.get(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending_records)

    def set(self, key: str, result: ComparisonResult) -> ComparisonResult:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot write to a closed comparison cache.")
            existing = self._records.get(key)
            if existing is not None:
                return existing

            self._records[key] = result
            self._pending_records[key] = result
            if self._should_flush():
                self._flush_pending()
            return result

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_pending()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_pending()
            self._connection.close()
            self._closed = True

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS comparisons (
                    key TEXT PRIMARY KEY,
                    reasoning TEXT NOT NULL,
                    is_ab_less_than_cd INTEGER NOT NULL CHECK (
                        is_ab_less_than_cd IN (0, 1)
                    ),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """)
            self._connection.commit()

    def _load_records(self) -> dict[str, ComparisonResult]:
        with self._lock:
            rows = self._connection.execute("""
                SELECT key, reasoning, is_ab_less_than_cd
                FROM comparisons
                """)
            return {
                row["key"]: ComparisonResult(
                    reasoning=row["reasoning"],
                    is_ab_less_than_cd=bool(row["is_ab_less_than_cd"]),
                )
                for row in rows
            }

    def _should_flush(self) -> bool:
        if not self._pending_records:
            return False
        if (
            self.flush_batch_size > 0
            and len(self._pending_records) >= self.flush_batch_size
        ):
            return True
        if self.sync_interval_seconds <= 0:
            return True
        return time.monotonic() - self._last_sync_at >= self.sync_interval_seconds

    def _flush_pending(self) -> None:
        if not self._pending_records:
            return

        self._connection.executemany(
            """
            INSERT OR IGNORE INTO comparisons (key, reasoning, is_ab_less_than_cd)
            VALUES (?, ?, ?)
            """,
            [
                (key, result.reasoning, int(result.is_ab_less_than_cd))
                for key, result in self._pending_records.items()
            ],
        )
        self._connection.commit()
        self._pending_records.clear()
        self._last_sync_at = time.monotonic()


@dataclass
class LLMDistanceComparator:
    """LLM-backed comparator for deciding whether d(a, b) < d(c, d)."""

    model: ChatModel
    temperature: float = 0.0
    max_tokens: int = 1024
    max_concurrency: int = 4
    max_batch_size: int = 1
    parse_retries: int = 2
    cache: ComparisonCache | None = field(default_factory=ComparisonCache)
    progress_callback: ComparisonProgressCallback | None = None
    _async_semaphores: weakref.WeakKeyDictionary[
        asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]
    ] = field(default_factory=weakref.WeakKeyDictionary, init=False, repr=False)

    def compare(self, a: str, b: str, c: str, d: str) -> ComparisonResult:
        cache = self.cache
        cache_key = comparison_cache_key(a, b, c, d)
        if cache is not None:
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                self._notify_progress("cached")
                return cached_result

        result = self._generate_and_parse(a, b, c, d)
        if cache is not None:
            result = cache.set(cache_key, result)
        self._notify_progress("generated")
        return result

    def compare_batch(
        self,
        comparisons: list[ComparisonInput],
        *,
        max_concurrency: int | None = None,
    ) -> list[ComparisonResult]:
        concurrency = (
            max_concurrency if max_concurrency is not None else self.max_concurrency
        )
        can_batch = self.max_batch_size > 1 and hasattr(
            self.model,
            "generate_batch_async",
        )
        if can_batch or (concurrency > 1 and hasattr(self.model, "generate_async")):
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
            cached_result = (
                self.cache.get(cache_key) if self.cache is not None else None
            )
            if cached_result is not None:
                results[index] = cached_result
                self._notify_progress("cached")
                continue

            pending = pending_by_key.get(cache_key)
            if pending is None:
                pending_by_key[cache_key] = (comparison, [index])
            else:
                pending[1].append(index)

        if pending_by_key:
            concurrency = (
                max_concurrency if max_concurrency is not None else self.max_concurrency
            )
            concurrency = max(1, concurrency)
            can_batch = self.max_batch_size > 1 and hasattr(
                self.model,
                "generate_batch_async",
            )
            if (
                concurrency == 1 or not hasattr(self.model, "generate_async")
            ) and not can_batch:
                for comparison, indexes in pending_by_key.values():
                    result = self.compare(*comparison)
                    for index in indexes:
                        results[index] = result
            else:
                semaphore = self._get_async_semaphore(concurrency)
                pending_items = list(pending_by_key.values())
                if can_batch:
                    tasks = [
                        (
                            asyncio.create_task(
                                self.compare_many_async(
                                    [comparison for comparison, _ in chunk],
                                    semaphore=semaphore,
                                )
                            ),
                            [indexes for _, indexes in chunk],
                        )
                        for chunk in _chunks(pending_items, self.max_batch_size)
                    ]
                    task_results = await asyncio.gather(*(task for task, _ in tasks))
                    for chunk_results, (_, chunk_indexes) in zip(task_results, tasks):
                        for result, indexes in zip(chunk_results, chunk_indexes):
                            for index in indexes:
                                results[index] = result
                else:
                    tasks = [
                        (
                            asyncio.create_task(
                                self.compare_async(*comparison, semaphore=semaphore)
                            ),
                            indexes,
                        )
                        for comparison, indexes in pending_items
                    ]
                    task_results = await asyncio.gather(*(task for task, _ in tasks))
                    for result, (_, indexes) in zip(task_results, tasks):
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
                self._notify_progress("cached")
                return cached_result

        if not hasattr(self.model, "generate_async"):
            return self.compare(a, b, c, d)

        result = await self._generate_and_parse_async(a, b, c, d, semaphore=semaphore)
        if cache is not None:
            result = cache.set(cache_key, result)
        self._notify_progress("generated")
        return result

    async def compare_many_async(
        self,
        comparisons: list[ComparisonInput],
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ComparisonResult]:
        if not comparisons:
            return []
        if not hasattr(self.model, "generate_batch_async"):
            return [
                await self.compare_async(*comparison, semaphore=semaphore)
                for comparison in comparisons
            ]

        results = await self._generate_and_parse_many_async(
            comparisons,
            semaphore=semaphore,
        )
        resolved_results: list[ComparisonResult] = []
        for comparison, result in zip(comparisons, results):
            if self.cache is not None:
                result = self.cache.set(comparison_cache_key(*comparison), result)
            self._notify_progress("generated")
            resolved_results.append(result)
        return resolved_results

    def _compare_batch_sequential(
        self,
        comparisons: list[ComparisonInput],
    ) -> list[ComparisonResult]:
        return [self.compare(*comparison) for comparison in comparisons]

    def _get_async_semaphore(self, concurrency: int) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        loop_semaphores = self._async_semaphores.get(loop)
        if loop_semaphores is None:
            loop_semaphores = {}
            self._async_semaphores[loop] = loop_semaphores

        semaphore = loop_semaphores.get(concurrency)
        if semaphore is None:
            semaphore = asyncio.Semaphore(concurrency)
            loop_semaphores[concurrency] = semaphore
        return semaphore

    def _notify_progress(self, event: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event)

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

    async def _generate_and_parse_many_async(
        self,
        comparisons: list[ComparisonInput],
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ComparisonResult]:
        messages_batch = [
            build_distance_comparison_messages(a, b, c, d) for a, b, c, d in comparisons
        ]
        generate_batch_async = getattr(self.model, "generate_batch_async")
        last_error: Exception | None = None
        for _ in range(self.parse_retries + 1):
            if semaphore is None:
                raw_responses = await generate_batch_async(
                    messages_batch,
                    response_format=COMPARISON_RESPONSE_FORMAT,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                async with semaphore:
                    raw_responses = await generate_batch_async(
                        messages_batch,
                        response_format=COMPARISON_RESPONSE_FORMAT,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

            if len(raw_responses) != len(comparisons):
                raise RuntimeError(
                    "Batch comparison generation did not produce every response."
                )

            try:
                return [
                    parse_comparison_response(raw_response)
                    for raw_response in raw_responses
                ]
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        if last_error is None:
            raise RuntimeError("Batch comparison generation failed without an error.")
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


def _chunks(
    items: list[tuple[ComparisonInput, list[int]]],
    size: int,
) -> list[list[tuple[ComparisonInput, list[int]]]]:
    chunk_size = max(1, size)
    return [
        items[start : start + chunk_size] for start in range(0, len(items), chunk_size)
    ]


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
    canonical_a, canonical_b = sorted((a, b))
    canonical_c, canonical_d = sorted((c, d))
    return json.dumps(
        {
            "version": COMPARISON_PROMPT_VERSION,
            "a": canonical_a,
            "b": canonical_b,
            "c": canonical_c,
            "d": canonical_d,
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
