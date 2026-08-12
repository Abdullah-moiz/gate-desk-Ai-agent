# Support SLA

## Response time targets (business hours, Mon-Fri)

| Tier | Channel | First response |
|---|---|---|
| Starter | Email | 48 hours |
| Pro | Email + chat | 24 hours |
| Business | Priority email + chat | 4 hours |
| Enterprise | Dedicated Slack + phone | 1 hour |

## Uptime guarantee
Business and Enterprise tiers are covered by a 99.9% monthly uptime guarantee. If uptime falls below this in a calendar month, affected customers are eligible for **service credits** (not cash refunds):

- 99.0-99.9% uptime: 10% credit
- 95.0-99.0% uptime: 25% credit
- Below 95.0% uptime: 50% credit

Credit requests must always be escalated to a human — automated agents cannot calculate or issue SLA credits, since it requires pulling incident data outside the support system.

## Bug severity & response
- **P0 (critical outage)**: Immediate escalation to engineering, regardless of tier. Never auto-close a P0.
- **P1 (major feature broken for a whole team or many users)**: Escalate within the tier's SLA window.
- **P2 (minor bug, workaround exists)**: Log and route to the engineering backlog; respond to the customer with the workaround if one exists.
- **P3 (cosmetic/low impact, affects one user)**: Log only, no escalation required.
