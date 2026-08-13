"""Phase 6: the confidence gate — the maker/checker split.

Phase 5's agent proposes an action; this module independently decides
whether that action is allowed to stand as an auto-resolution, or must be
force-escalated instead. The agent's original tool call is never discarded
when overridden — it's preserved as a draft (see app/agent.py) so a human
reviewing the escalation queue can see what the agent would have done.

Two kinds of checks, deliberately kept separate:
  1. Deterministic policy rules (e.g. P0/P1 bugs always escalate) — no
     confidence score can override these.
  2. Probabilistic confidence thresholds (retrieval grounding + the agent's
     own self-reported confidence) — these catch the cases no hard-coded
     rule anticipated, at the cost of being calibrated, not exact.
"""

from dataclasses import dataclass

# Actions with a real (if mocked) external side effect get a stricter
# confidence bar than purely informational/logging actions.
HIGH_RISK_ACTIONS = {"issue_refund", "send_password_reset", "unlock_account"}
LOW_RISK_ACTIONS = {"log_bug", "provide_workaround", "answer_faq", "acknowledge", "request_more_info", "close_spam"}

# Dense cosine similarity of the top retrieved policy chunk. Below this, the
# agent's decision isn't actually grounded in retrieved policy text no matter
# how confident it claims to be — RRF fusion rank scores aren't usable here
# since they reflect rank, not semantic relevance (see app/retrieval.py).
# Calibrated empirically: well-grounded queries scored ~0.63-0.66, a fully
# unrelated query scored 0.14 — 0.35 sits in the gap with margin either side.
POLICY_CONFIDENCE_THRESHOLD = 0.35

AGENT_CONFIDENCE_THRESHOLD = {
    "high_risk": 0.85,
    "low_risk": 0.60,
}


@dataclass
class GateDecision:
    passed: bool
    reason: str
    final_outcome: str  # "auto_resolved" | "escalated"
    final_action: str
    final_arguments: dict
    final_result: dict


def apply_gate(
    action: str,
    arguments: dict,
    result: dict,
    agent_confidence: float | None,
    policy_confidence: float,
) -> GateDecision:
    if action == "escalate_to_human":
        return GateDecision(True, "agent chose to escalate", "escalated", action, arguments, result)

    if action == "log_bug" and arguments.get("severity") in ("P0", "P1"):
        reason = (
            f"{arguments.get('severity')} bugs must always be escalated to engineering per "
            "policy, regardless of the agent's tool choice."
        )
        return _escalate(reason)

    risk_tier = "high_risk" if action in HIGH_RISK_ACTIONS else "low_risk"

    # Policy grounding is only load-bearing for high-risk actions. A weak
    # top similarity on a low-risk action (acknowledge, log a P3 bug, ...)
    # usually just means the retrieved policy had nothing to say — because
    # the action doesn't need policy backing in the first place, not
    # because the agent is ungrounded. Gating those on retrieval score was
    # tried and over-escalated correct, low-consequence resolutions.
    if risk_tier == "high_risk" and policy_confidence < POLICY_CONFIDENCE_THRESHOLD:
        reason = f"policy grounding too weak (top similarity {policy_confidence:.2f} < {POLICY_CONFIDENCE_THRESHOLD})"
        return _escalate(reason)

    threshold = AGENT_CONFIDENCE_THRESHOLD[risk_tier]
    if agent_confidence is None or agent_confidence < threshold:
        reason = (
            f"agent confidence {agent_confidence} below the {threshold} bar required for "
            f"{risk_tier.replace('_', ' ')} action '{action}'"
        )
        return _escalate(reason)

    return GateDecision(True, "passed all gate checks", "auto_resolved", action, arguments, result)


def _escalate(reason: str) -> GateDecision:
    args = {"reason": f"Gate override: {reason}"}
    result = {"escalated": True, "reason": args["reason"]}
    return GateDecision(False, reason, "escalated", "escalate_to_human", args, result)
