"""Tracks OpenAI token usage across a process run, for the cost line in
Phase 8's eval report. Handles both API shapes in this project: Chat
Completions usage (prompt_tokens/completion_tokens, used by app.routing's
classify()) and Responses API usage (input_tokens/output_tokens, used by
app.agent's tool-calling loop).
"""

# $ per 1M tokens. Only models actually used in this project need an entry.
PRICING = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}

_totals: dict[str, dict[str, int]] = {}


def record(model: str, usage) -> None:
    if usage is None:
        return
    bucket = _totals.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", 0)
    bucket["input_tokens"] += input_tokens or 0
    bucket["output_tokens"] += output_tokens or 0
    bucket["calls"] += 1


def reset() -> None:
    _totals.clear()


def summary() -> tuple[list[dict], float]:
    """Returns (per-model rows, total estimated cost in USD)."""
    rows = []
    total_cost = 0.0
    for model, bucket in _totals.items():
        price = PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = bucket["input_tokens"] / 1_000_000 * price["input"] + bucket["output_tokens"] / 1_000_000 * price["output"]
        total_cost += cost
        rows.append({"model": model, **bucket, "estimated_cost_usd": cost})
    return rows, total_cost
