# Evaluation Report

Ran 16 tickets from `data/eval_set.json` through the full pipeline 
(classify -> retrieve -> agent tool-calling loop -> confidence gate).

| Metric | Value |
|---|---|
| Category classification accuracy | 100.0% |
| Outcome accuracy (auto_resolved vs escalated) | 100.0% |
| Action accuracy (on correctly auto-resolved tickets) | 100.0% |
| **False-resolve rate** (wrongly auto-resolved when it should've escalated) | **0.0%** (0 tickets) |
| Over-escalation rate (escalated when it could've auto-resolved) | 0.0% (0 tickets) |
| Policy grounding hit rate (proxy retrieval metric, see script docstring) | 100.0% |
| Estimated cost for this eval run | $0.00978 ($0.000611/ticket) |

## Token usage by model

| Model | Calls | Input tokens | Output tokens | Est. cost |
|---|---|---|---|---|
| gpt-5-nano | 46 | 44,621 | 18,875 | $0.00978 |

## Per-ticket results

| Ticket | Category | Outcome | Action | Grounded | Gate |
|---|---|---|---|---|---|
| E-01 | OK | OK | OK | OK | passed |
| E-02 | OK | OK | OK | OK | passed |
| E-03 | OK | OK | OK | OK | passed |
| E-04 | OK | OK | OK | OK | passed |
| E-05 | OK | OK | OK | OK | passed |
| E-06 | OK | OK | OK | OK | passed |
| E-07 | OK | OK | OK | OK | passed |
| E-08 | OK | OK | OK | OK | passed |
| E-09 | OK | OK | OK | OK | passed |
| E-10 | OK | OK | OK | OK | passed |
| E-11 | OK | OK | OK | OK | passed |
| E-12 | OK | OK | OK | OK | passed |
| E-13 | OK | OK | OK | OK | passed |
| E-14 | OK | OK | OK | OK | passed |
| E-15 | OK | OK | OK | OK | passed |
| E-16 | OK | OK | OK | OK | passed |
