"""app.py

Streamlit UI for the NetSoft 2026 network assistant (demo version).

This UI is intentionally simple for interview / demo use:
- Uses a single model (qwen2.5:14b-instruct)
- No benchmarking panel
- No prompt-mode / RAG toggles
- Always shows the model/Python JSON output
- Shows plot if provided
- Shows tables (when present)
"""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st

import llm_agent


# --------------------------- App config ---------------------------
st.set_page_config(page_title="Network Assistant", page_icon="🛜", layout="wide")
st.title("🛜 LLM-based Network Assistant")
#st.caption("Hybrid execution: Python computes graph metrics/equipment summaries; model is used as a strict-JSON assistant when needed.")

# Model selector (Ollama)
st.sidebar.header("Model")
model = st.sidebar.selectbox(
    "Choose an Ollama model",
    options=sorted(list(llm_agent.VALID_MODELS)),
    index=sorted(list(llm_agent.VALID_MODELS)).index(llm_agent.get_model()) if hasattr(llm_agent, "get_model") else 0,
)
try:
    llm_agent.set_model(model)
except Exception as e:
    st.sidebar.error(str(e))

st.sidebar.caption("Tip: run `ollama pull <model>` if a model isn't installed.")


#st.sidebar.header("ℹ️ Demo settings")
#st.sidebar.write(f"Model: **{llm_agent.get_model()}**")
#st.sidebar.caption("To change the model later, edit app.py (kept simple on purpose).")

#st.divider()

# --------------------------- Interactive panel ---------------------------
st.subheader("Ask a network question")
query = st.text_input("Enter your question")
go = st.button("Run")

if go and query.strip():
    # Reset per question to avoid cross-query contamination in demos
    llm_agent._reset_state()

    out = llm_agent.run(query)

    llm_json = out.get("llm_json") if isinstance(out, dict) else None
    if not isinstance(llm_json, dict):
        st.error("No JSON produced.")
        st.code(out.get("raw", ""))
    else:
        # Plot
        if out.get("fig") is not None:
            st.write("### Plot")
            st.pyplot(out["fig"], clear_figure=False)

        # Tables
        node_df = out.get("node_df")
        link_df = out.get("link_df")
        qot_df = out.get("qot_df")

        if isinstance(node_df, pd.DataFrame) and not node_df.empty:
            st.write("### Table")
            st.dataframe(node_df, use_container_width=True)

        if isinstance(link_df, pd.DataFrame) and not link_df.empty:
            st.write("### Table")
            st.dataframe(link_df, use_container_width=True)

        if isinstance(qot_df, pd.DataFrame) and not qot_df.empty:
            st.write("### Table")
            st.dataframe(qot_df, use_container_width=True)

        # Raw JSON always visible
        st.write("### LLM JSON Output")
        st.code(json.dumps(llm_json, indent=2), language="json")


st.divider()
st.subheader("✅ Test suite (your 33 questions)")
st.caption("Runs each question once and shows whether the backend returned an error. This is useful to quickly compare Ollama models.")

DEFAULT_QUESTIONS = [
    "Create a topology from CSV files and visualize the full network topology.",
    "Create a topology from CSV files (no visualization).",
    "Draw the network topology from the CSV files and label each link with its distance.",
    "Find the shortest path from Berlin to Munich and plot that path only.",
    "Draw the shortest path from Berlin to Stuttgart and plot that path only.",
    "Show the longest path from Berlin to Stuttgart by total distance and return hop count and total distance.",
    "Among all paths between Berlin and Stuttgart, which path has the fewest hops? Return the path and hop count.",
    "Which node has the highest degree in the current topology? Return the node id and degree.",
    "Which node has the lowest degree in the current topology? Return the node id and degree.",
    "Compute the average node degree, the minimum degree, and the maximum degree.",
    "Count how many nodes have degree 1.",
    "Count how many nodes have degree ≥ 3.",
    "Is there any isolated node (degree 0)? Return true or false.",
    "Find the top 3 nodes by degree and return them in descending order.",
    "Compute the average shortest-path length of the topology in hops.",
    "Show ML links under 5 km and return a table (no plot).",
    "Show AL links under 10 km and return a table (no plot).",
    "Calculate the average ML link distance in km.",
    "Calculate the average AL link distance in km.",
    "Find the maximum ML link distance and plot only that edge highlighted.",
    "Return the 5 longest links in the topology with endpoints, link_id, and distance_km.",
    "Deploy network equipment and then show a summary table for node equipment (no plot).",
    "After deploying equipment, how many ROADMs were placed in total?",
    "After deploying equipment, how many TRx elements were placed in total?",
    "After deploying equipment, count the total number of ILAs across all links.",
    "After deploying equipment, list all links that have an ILA deployed (link endpoints and equipment list).",
    "After deploying equipment, list all links that have a PreAmp deployed (link endpoints and equipment list).",
    "After deploying equipment, list all links that have an Amp deployed (link endpoints and equipment list).",
    "Propagation delay from Berlin to Munich.",
    "Which fiber type gives the minimum propagation delay from Berlin to Stuttgart?",
    "If we have to deploy HCF on only 2 links in the topology, what are the best links to minimize propagation delay?",
    "From Berlin to Stuttgart, what possible route has the minimum number of ILAs?",
    "From Berlin to Stuttgart, which links have ILAs ≥ 4? List all possible paths that include such links",
]

with st.expander("Edit the test questions"):
    qs_text = st.text_area("One question per line", value="\n".join(DEFAULT_QUESTIONS), height=260)
    test_questions = [q.strip() for q in qs_text.splitlines() if q.strip()]

run_suite = st.button("Run test suite")
if run_suite:
    rows = []
    for i, q in enumerate(test_questions, start=1):
        llm_agent._reset_state()
        out = llm_agent.run(q)
        jj = out.get("llm_json") if isinstance(out, dict) else {}
        err = (jj.get("result") or {}).get("error")
        rows.append({"#": i, "question": q, "error": err or ""})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    n_err = int((df["error"].astype(str).str.len() > 0).sum())
    if n_err == 0:
        st.success("All tests returned without errors.")
    else:
        st.warning(f"{n_err} tests returned errors. Click a row and run it in the single-question panel to debug.")
