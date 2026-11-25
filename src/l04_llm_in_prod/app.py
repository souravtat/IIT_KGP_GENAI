"""
Walmart Support Bot — Building LLM Systems hands-on (Flex Returns pilot scenario).

Every component is a toggle in the sidebar. Hold the query fixed, flip one switch,
read the pipeline table. This file is the UI and the ordering; the parts are in
components.py.

    streamlit run app.py

Environment:
    OPENAI_API_KEY   your issued key
    OPENAI_BASE_URL  your gateway endpoint (optional)
"""

import re
import json
import time
import streamlit as st
import components as c

SYSTEM = ("You are a Walmart associate support assistant for the Flex Returns pilot. "
          "Only help with Flex Returns support questions (returns, refunds, orders, "
          "policy). If asked about anything else, politely decline and say you can only "
          "help with Flex Returns support questions. "
          "Answer concisely and only address what the customer actually asked. Do not "
          "proactively volunteer unrelated procedural details (e.g. RA code requirements, "
          "escalation steps) unless the customer is asking about starting a return or refund.")

TOOL_INSTRUCTIONS = (
    "\n\nYou may call tools by replying with a line of the form "
    "CALL: get_order(W-1234), CALL: issue_refund(W-1234) or CALL: send_email(W-1234). "
    "Look the order up if you need its status or details. Check the return window "
    "and the approval threshold in the policy before issuing a refund."
)


def lookup_orders(working, mapping):
    """Finds order IDs in the (possibly masked) query and resolves each to its
    record. Returns [(order_id, record), ...] for whatever's found."""
    found = []
    for token in re.findall(r"\[ORDER_\d+\]|\bW-\d{4}\b", working):
        order_id = mapping.get(token, token)
        order = c.get_order(order_id)
        if order:
            found.append((order_id, order))
    return found


def format_history(messages, limit=6):
    """Plain-text transcript of the last `limit` prior messages (~3 exchanges),
    for conversational continuity. Capped so the prompt doesn't grow unboundedly
    over a long chat."""
    lines = []
    for msg in messages[-limit:]:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def fmt_ms(seconds):
    """Sub-millisecond steps (in-memory dict lookups, no network/DB call) round
    to 0 -- that's a real measurement, not a bug, so show "<1" instead of a
    misleading "0"."""
    ms = round(seconds * 1000)
    return ms if ms else "<1"


def run(query, cfg, log, history_text=""):
    """The whole system. Each block is one step of the session.

    `history_text` is a plain-text transcript of recent prior turns (see
    format_history below), for conversational continuity -- e.g. "refund it"
    resolving against an order ID mentioned two turns ago. It only ever folds
    into the prompt text; it never touches cache keys (exact/semantic cache
    still key on the raw current query only, so a follow-up doesn't collide
    with an unrelated earlier turn's cached answer)."""

    def step(name, fn, *args, **kw):
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        log.append({"step": name, "ms": fmt_ms(time.perf_counter() - t0),
                    "result": ""})
        return out

    # 5a: exact cache
    if cfg["exact_cache"]:
        hit = step("exact_cache", c.exact_lookup, query)
        log[-1]["result"] = "HIT" if hit else "miss"
        if hit:
            return hit

    # 5b: semantic cache
    if cfg["semantic_cache"]:
        hit, detail = step("semantic_cache", c.semantic_lookup, query, cfg["threshold"])
        log[-1]["result"] = (f"{'HIT' if hit else 'miss'} (closest {detail[0]!r} @ {detail[1]})"
                             if detail else "miss (empty)")
        if hit:
            return hit

    # 3: PII guardrail, input side -- reversible masking
    mapping, working = {}, query
    if cfg["guardrails_pii"]:
        working, mapping = step("mask", c.mask, query)
        log[-1]["result"] = json.dumps(mapping) if mapping else "nothing masked"

    # 4: routing
    model = c.BIG_MODEL
    if cfg["router"]:
        label = step("route", c.route, working)
        log[-1]["result"] = label
        if label == "out_of_scope":
            return "I can only help with Flex Returns support questions."
        model = c.SMALL_MODEL if label == "simple" else c.BIG_MODEL

    # 2: context construction
    system = SYSTEM
    if history_text:
        system += ("\n\nRecent conversation so far, for context only -- the customer's "
                   "latest message below is the actual question to answer:\n" + history_text)
    if cfg["context"]:
        chunks = step("retrieve", c.retrieve, working)
        log[-1]["result"] = ", ".join(ch.split("\n")[0][3:] for ch in chunks)
        system += ("\n\nAnswer only from the policy below and any order record provided. "
                    "If neither covers the question, say so.\n\nPolicy:\n" + "\n\n".join(chunks))

        # Read action: an order ID in the query is looked up and injected into
        # context automatically. This is silent by design (spec step 2/6) -- the
        # model never calls this as a tool, so the same channel later carries the
        # planted-injection demo in step 6. Only logged when it actually finds an
        # order, same as the CALL: tools below only log when actually called.
        t0 = time.perf_counter()
        orders_found = lookup_orders(working, mapping)
        if not orders_found and history_text:
            # "refund it" has no order ID of its own -- fall back to the ID
            # mentioned in a prior turn, so follow-ups resolve correctly.
            orders_found = lookup_orders(history_text, {})
        if orders_found:
            log.append({"step": "get_order",
                        "ms": fmt_ms(time.perf_counter() - t0),
                        "result": ", ".join(oid for oid, _ in orders_found)})
        for order_id, order in orders_found:
            system += f"\n\nAuthoritative order record for {order_id}: {json.dumps(order)}"
        if not orders_found:
            system += ("\n\nIf the question concerns a specific order but no order ID "
                       "(format W-####) was given, do not say you lack access to order "
                       "information -- simply ask the customer for their order ID.")

    if cfg["actions"]:
        system += TOOL_INSTRUCTIONS

    answer, tokens = step("generate", c.llm, system, working, model)
    log[-1]["result"] = f"{model}, {tokens} tokens"

    # 6: write actions
    if cfg["actions"]:
        for name, arg in re.findall(r"CALL:\s*(\w+)\(([^)]*)\)", answer):
            if name in c.TOOLS:
                result = step(f"tool:{name}", c.TOOLS[name], arg.strip())
                log[-1]["result"] = result
                answer += f"\n\n[{result}]"

    # 3: PII guardrail -- reversible mask on the way in, restore on the way out.
    # Scoped to the user's own query only: PII that shows up in the answer
    # because it came from retrieved policy/context (not from what the customer
    # typed) is not scanned for or blocked here.
    if cfg["guardrails_pii"]:
        answer = step("unmask", c.unmask, answer, mapping)
        log[-1]["result"] = f"{len(mapping)} restored"

    if cfg["exact_cache"] or cfg["semantic_cache"]:
        step("cache_store", c.cache_store, query, answer)

    return answer


st.set_page_config(page_title="Walmart Support Bot", layout="wide")

with st.sidebar:
    st.header("Pipeline")
    cfg = {
        "context": st.toggle("2. Context (RAG + order lookup)"),
        "guardrails_pii": st.toggle("3. PII guardrail (reversible mask)"),
        "router": st.toggle("4. Router"),
        "exact_cache": st.toggle("5a. Exact cache"),
        "semantic_cache": st.toggle("5b. Semantic cache"),
        "actions": st.toggle("6. Write actions"),
    }
    cfg["threshold"] = st.slider("Semantic threshold", 0.70, 1.00, 0.95, 0.01)

    st.divider()
    st.caption(f"Cached: {c.cache_stats()}")
    if st.button("Clear caches"):
        c.cache_clear()
        st.rerun()
    if st.button("Plant injection in W-1044"):
        c.plant_injection()
    if st.button("Reset orders"):
        c.reset_orders()
        st.rerun()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("Walmart Support Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {role, content, ms, log}

# Replay history first. Each entry carries its own frozen pipeline log, so
# flipping a sidebar toggle mid-conversation never rewrites an earlier turn's
# table -- only the next `run()` call sees the new cfg.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("log") is not None:
            st.caption(f"Total latency: {msg['ms']} ms")
            st.dataframe(msg["log"], use_container_width=True)

query = st.chat_input("Ask your support question...")
if query:
    history_text = format_history(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": query, "ms": None, "log": None})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            log = []
            t0 = time.perf_counter()
            answer = run(query, cfg, log, history_text)
            ms = round((time.perf_counter() - t0) * 1000)
        st.markdown(answer)
        st.caption(f"Total latency: {ms} ms")
        st.dataframe(log, use_container_width=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "ms": ms, "log": log})

with st.expander("Orders"):
    st.json(c.ORDERS)
