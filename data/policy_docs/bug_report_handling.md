# Bug Report Handling

When a customer reports a bug:

1. Try to match it against a similar past ticket. If one exists with a known workaround, share it immediately rather than re-investigating from scratch.
2. Classify severity using the P0–P3 scale (see sla.md).
3. **P0/P1 bugs are always escalated to engineering.** An automated agent should never mark these as resolved on its own, even if a temporary workaround is offered — the underlying defect still needs a fix, and these affect either data integrity or many users at once.
4. **P2/P3 bugs can be closed with a workaround** plus a note that it's logged in the engineering backlog; auto-resolution is acceptable here.
5. If the report doesn't include enough detail to classify severity (no steps to reproduce, unclear scope), ask a clarifying question rather than guessing — this is a safe auto-handled action since it takes no irreversible action.
6. Always try to capture: browser/OS, steps to reproduce, screenshot/video if available, and whether it affects the whole team or just one user — this last point is often the fastest signal for P1 vs P2/P3.
