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
st.caption("Hybrid execution: Python computes graph metrics/equipment summaries; model is used as a strict-JSON assistant when needed.")

# Force single model for this demo app
try:
    llm_agent.set_model("qwen2.5:14b-instruct")
except Exception:
    pass

st.sidebar.header("ℹ️ Demo settings")
st.sidebar.write(f"Model: **{llm_agent.get_model()}**")
st.sidebar.caption("To change the model later, edit app.py (kept simple on purpose).")

st.divider()

# --------------------------- Interactive panel ---------------------------
st.subheader("Ask a network question")
query = st.text_input("Enter your question", value="Create a topology from CSV files and visualize the full network topology.")
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
        st.write("### Raw JSON")
        st.code(json.dumps(llm_json, indent=2), language="json")
