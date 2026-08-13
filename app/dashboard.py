"""Phase 9/10: the demo dashboard. A pure HTTP client of app/main.py's API
— it holds no OpenAI/Voyage/Qdrant/Postgres credentials itself, only
API_BASE_URL. That split is what makes the two docker-compose services
(api, dashboard) meaningfully separate rather than the same code twice.

Two views: a ticket queue backed by GET /tickets + GET /tickets/{id}, and
a live "submit a new ticket" form that POSTs to the API and renders the
result while you watch, for demos/interviews.

Run with: streamlit run app/dashboard.py
(reads API_BASE_URL from the environment, defaults to http://localhost:8000)
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="GateDesk", page_icon="🎫", layout="wide")

OUTCOME_COLOR = {"auto_resolved": "green", "escalated": "orange"}


def api_get(path: str):
    resp = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def render_ticket_detail(ticket_id: str) -> None:
    try:
        detail = api_get(f"/tickets/{ticket_id}")
    except requests.HTTPError as e:
        st.warning(f"No ticket found with id '{ticket_id}' ({e}).")
        return

    ticket = detail["ticket"]
    st.subheader(f"{ticket_id} — {ticket['subject']}")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Category", ticket["category"] or "?")
    meta_cols[1].metric("Urgency", ticket["urgency"] or "?")
    meta_cols[2].metric("Customer", ticket["customer_email"] or "unknown")
    meta_cols[3].metric("Created", str(ticket["created_at"])[:19])
    st.text_area("Body", ticket["body"], height=80, disabled=True, key=f"body-{ticket_id}")

    res = detail["resolution"]
    if res:
        outcome_color = OUTCOME_COLOR.get(res["outcome"], "gray")
        st.markdown(f"### Resolution: :{outcome_color}[{res['outcome'].upper()}] — `{res['action']}`")

        gate_cols = st.columns(4)
        gate_cols[0].metric("Agent confidence", res["agent_confidence"] if res["agent_confidence"] is not None else "n/a")
        gate_cols[1].metric("Policy grounding", f"{res['policy_confidence']:.2f}")
        gate_cols[2].metric("Precedent grounding", f"{res['precedent_confidence']:.2f}")
        gate_cols[3].metric("Gate", "PASSED" if res["gate_passed"] else "OVERRIDDEN")

        if not res["gate_passed"]:
            st.warning(
                f"**Gate overrode the agent's draft.** Agent wanted to call "
                f"`{res['agent_action']}({res['agent_arguments']})` — reason: {res['gate_reason']}"
            )

        with st.expander("Final result payload"):
            st.json(res["result"])

    tab_retrieval, tab_tools = st.tabs(["Retrieval", "Tool calls"])

    with tab_retrieval:
        retrievals = detail["retrievals"]
        if not retrievals:
            st.caption("No retrieval logged for this ticket.")
        else:
            col_policy, col_precedent = st.columns(2)
            for r in retrievals:
                target = col_policy if r["collection_name"] == "policy_kb" else col_precedent
                payload = r["payload"]
                if r["collection_name"] == "policy_kb":
                    label = f"**[{r['result_rank']}] score={r['score']:.3f}** — {payload['doc_type']} / {payload['section']}"
                    body = payload["text"].split("\n", 1)[-1]
                else:
                    resolution = payload["resolution"]
                    label = f"**[{r['result_rank']}] score={r['score']:.3f}** — {payload['id']} \"{payload['subject']}\""
                    body = f"{payload['body']}\n\n→ {resolution['outcome']} via {resolution['action']} ({resolution['reason']})"
                target.markdown(label)
                target.caption(body)
            col_policy.caption("policy_kb")
            col_precedent.caption("resolved_tickets (precedent)")

    with tab_tools:
        tool_calls = detail["tool_calls"]
        if not tool_calls:
            st.caption("No tool calls logged for this ticket.")
        for tc in tool_calls:
            st.markdown(f"**[{tc['step']}] `{tc['tool_name']}`**")
            arg_col, res_col = st.columns(2)
            arg_col.json(tc["arguments"])
            res_col.json(tc["result"])


def render_queue() -> None:
    try:
        tickets = api_get("/tickets")
    except requests.RequestException as e:
        st.error(f"Couldn't reach the API at {API_BASE_URL} ({e}). Is the api service running?")
        return

    if not tickets:
        st.info("No tickets yet. Submit one in the other tab, or run `python scripts/evaluate.py` to populate demo data.")
        return

    outcome_filter = st.radio("Filter", ["All", "Auto-resolved", "Escalated"], horizontal=True)
    if outcome_filter == "Auto-resolved":
        tickets = [t for t in tickets if t["outcome"] == "auto_resolved"]
    elif outcome_filter == "Escalated":
        tickets = [t for t in tickets if t["outcome"] == "escalated"]

    st.dataframe(
        [
            {
                "Ticket": t["ticket_id"],
                "Subject": t["subject"],
                "Category": t["category"],
                "Urgency": t["urgency"],
                "Outcome": t["outcome"],
                "Action": t["action"],
            }
            for t in tickets
        ],
        use_container_width=True,
        hide_index=True,
    )

    selected = st.selectbox("Inspect a ticket", [t["ticket_id"] for t in tickets])
    if selected:
        st.divider()
        render_ticket_detail(selected)


def render_submit() -> None:
    st.caption(
        "Runs the real pipeline live via the API: classify -> retrieve -> agent tool-calling loop -> "
        "confidence gate. Use a customer email from `data/accounts_seed.json` (e.g. alice@acme.io) so "
        "account lookups resolve."
    )
    with st.form("submit_ticket"):
        subject = st.text_input("Subject")
        body = st.text_area("Body", height=120)
        customer_email = st.text_input("Customer email", placeholder="alice@acme.io")
        submitted = st.form_submit_button("Submit & run agent")

    if not submitted:
        return
    if not subject or not body:
        st.error("Subject and body are required.")
        return

    with st.spinner("Classifying, retrieving, running the agent, and checking the gate..."):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/tickets",
                json={"subject": subject, "body": body, "customer_email": customer_email or None},
                timeout=120,
            )
            resp.raise_for_status()
        except requests.HTTPError as e:
            detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
            st.error(detail)
            return
        except requests.RequestException as e:
            st.error(f"Couldn't reach the API at {API_BASE_URL} ({e}). Is the api service running?")
            return

    ticket_id = resp.json()["ticket_id"]
    st.success(f"Ticket {ticket_id} processed — see below.")
    render_ticket_detail(ticket_id)


st.title("🎫 GateDesk")
st.caption(
    "RAG + agentic support triage for Flowbase, with a confidence gate as the safety layer between "
    "the agent's tool choice and an actual auto-resolution."
)

tab_queue, tab_submit = st.tabs(["Ticket Queue", "Submit New Ticket"])
with tab_queue:
    render_queue()
with tab_submit:
    render_submit()
