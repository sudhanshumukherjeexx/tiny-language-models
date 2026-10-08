"""Optional qualitative evaluation: an LLM judge or a manual-rating export.

Nothing in the core pipeline depends on this module. Without an API key the
generations are exported as a CSV rating sheet for human raters, using the
same rubric. With ``ANTHROPIC_API_KEY`` set (and ``uv sync --extra judge``),
``AnthropicJudge`` scores each story and every artifact needed to audit the
judgement is saved: the exact prompt, the provider/model that actually
served the request, the raw response, and the parsed scores.

Caveats, stated rather than hidden: an LLM judge is itself a model with
biases (e.g. toward longer or more polished text), and current Claude models
do not accept sampling parameters, so repeated judgements of the same story
can differ. Treat scores as a noisy, reviewable signal, and report their
spread across stories, not single numbers.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

RUBRIC: dict[str, str] = {
    "grammar": "Spelling, grammar and punctuation are correct.",
    "coherence": "Sentences follow logically; the story makes sense as a whole.",
    "consistency": "Characters, objects and facts stay consistent; no contradictions.",
    "creativity": "The story contains some original or surprising element.",
    "completeness": "The story reaches a sensible ending rather than stopping abruptly.",
    "repetition": "Free of repeated sentences or loops (5 = no problematic repetition).",
}

JUDGE_PROMPT = """You are grading a short children's story written by a small language model \
trained only on simple children's stories. The model was given the prompt below and \
continued it.

Score the CONTINUATION from 1 (very poor) to 5 (excellent) on each criterion:
{criteria}

Judge only what is written. Do not reward length for its own sake. A continuation that \
was cut off by a length limit should lose points on completeness only.

PROMPT:
<prompt>{prompt}</prompt>

CONTINUATION:
<continuation>{continuation}</continuation>

Return the scores and a one-sentence justification per criterion."""

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        name: {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                "justification": {"type": "string"},
            },
            "required": ["score", "justification"],
            "additionalProperties": False,
        }
        for name in RUBRIC
    },
    "required": list(RUBRIC),
    "additionalProperties": False,
}


def build_prompt(prompt: str, continuation: str) -> str:
    criteria = "\n".join(f"- {k}: {v}" for k, v in RUBRIC.items())
    return JUDGE_PROMPT.format(criteria=criteria, prompt=prompt, continuation=continuation)


@dataclass
class JudgeResult:
    item_id: str
    provider: str
    requested_model: str
    served_model: str | None
    prompt: str
    raw_response: str | None
    scores: dict[str, int] = field(default_factory=dict)
    justifications: dict[str, str] = field(default_factory=dict)
    stop_reason: str | None = None
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Judge(Protocol):
    provider: str
    model: str

    def score(self, item_id: str, prompt: str, continuation: str) -> JudgeResult: ...


class AnthropicJudge:
    """Claude-based judge using JSON-schema structured output."""

    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium"):
        try:
            import anthropic
        except ImportError as err:
            raise ImportError("Install the judge extra: uv sync --extra judge") from err
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # resolves credentials from the environment
        self.model = model
        self.effort = effort

    def score(self, item_id: str, prompt: str, continuation: str) -> JudgeResult:
        anthropic = self._anthropic
        text = build_prompt(prompt, continuation)
        result = JudgeResult(item_id, self.provider, self.model, None, text, None)
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                messages=[{"role": "user", "content": text}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": SCORE_SCHEMA},
                },
                # On a safety decline the API re-runs the request on a fallback
                # model; the serving model is recorded in served_model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as err:
            result.error = f"rate_limited: {err.message}"
            return result
        except anthropic.APIStatusError as err:
            result.error = f"api_error {err.status_code}: {err.message}"
            return result
        except anthropic.APIConnectionError as err:
            result.error = f"connection_error: {err}"
            return result
        result.served_model = response.model
        result.stop_reason = response.stop_reason
        if response.stop_reason == "refusal":
            result.error = "refused"
            return result
        raw = next((b.text for b in response.content if b.type == "text"), None)
        result.raw_response = raw
        if raw is None:
            result.error = "no text block in response"
            return result
        parsed = json.loads(raw)
        result.scores = {k: int(v["score"]) for k, v in parsed.items()}
        result.justifications = {k: v["justification"] for k, v in parsed.items()}
        return result


def judge_records(judge: Judge, records: list[dict[str, Any]]) -> list[JudgeResult]:
    return [
        judge.score(f"{r['prompt_id']}|{r['preset']}|{r['seed']}", r["prompt"], r["completion"])
        for r in records
    ]


def aggregate(results: list[JudgeResult]) -> dict[str, Any]:
    ok = [r for r in results if r.scores]
    out: dict[str, Any] = {"scored": len(ok), "failed": len(results) - len(ok)}
    for name in RUBRIC:
        vals = [r.scores[name] for r in ok if name in r.scores]
        if vals:
            out[name] = {
                "mean": statistics.fmean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    return out


def export_manual_sheet(records: list[dict[str, Any]], path: str | Path) -> Path:
    """CSV with one row per generation and an empty column per rubric criterion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "category", "prompt", "continuation", *RUBRIC, "notes"])
        for r in records:
            writer.writerow(
                [
                    f"{r['prompt_id']}|{r['preset']}|{r['seed']}",
                    r["category"],
                    r["prompt"],
                    r["completion"],
                    *([""] * len(RUBRIC)),
                    "",
                ]
            )
    return path


def judge_available() -> bool:
    """True when an Anthropic credential is configured in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
