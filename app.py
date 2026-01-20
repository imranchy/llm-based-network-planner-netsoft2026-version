"""app.py

Streamlit UI for the NetSoft 2026 LLM-first network assistant.

Key properties:
- The LLM returns ONE JSON object (no python calls).
- Graph analytics (paths, degrees, hop count, etc.) are computed by the LLM.
- Python is used for: topology building, QoT reference calculations, equipment summaries, and plotting.
"""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from llm_agent import (
    VALID_MODELS,
    set_model,
    get_model,
    run,
    benchmark_models,
    load_benchmark_queries_from_yaml,
    _reset_state,
)


st.set_page_config(page_title="LLM-based Network Assistant", page_icon="🛜", layout="wide")
st.title("🛜 LLM-based Network Assistant")
st.caption("LLM-first analytics: the model computes paths/metrics from a topology snapshot and returns strict JSON.")

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    available_models = sorted(list(VALID_MODELS))
    active_model = st.selectbox("Active Model", available_models, index=available_models.index(get_model()) if get_model() in available_models else 0)
    set_model(active_model)
    st.caption(f"Active model: **{get_model()}**")

    st.markdown("---")
    st.subheader("Prompting")
    prompt_mode = st.radio("Prompting mode", ["none", "zero-shot", "one-shot", "few-shot"], index=1)
    rag_enabled = st.toggle("Enable RAG", value=False, help="Adds equipment/QoT reference tables + optional rag_corpus/* notes to the model context")

    st.markdown("---")
    st.subheader("📈 Benchmark")
    bench_models = st.multiselect("Models to compare", available_models, default=[active_model])

    query_source = st.radio("Query source", ["Defaults", "All YAML queries", "Custom"], index=0)

    default_queries = [
        "Create a topology from CSV files.",
        "Compute the shortest path from M1 to M3 and provide hop count.",
        "Which node has the highest degree? Provide the degree value.",
        "What is the propagation delay between M1 and M3 (assume speed of light in fiber = 2e8 m/s)?",
    ]

    if query_source == "Defaults":
        query_text = st.text_area("Benchmark queries", value="\n".join(default_queries), height=160)
        def get_queries():
            return [q.strip() for q in query_text.splitlines() if q.strip()]

    elif query_source == "All YAML queries":
        st.caption("Using templates from questions.yaml")
        def get_queries():
            return load_benchmark_queries_from_yaml("questions.yaml")

    else:
        custom_text = st.text_area("Benchmark queries", height=160)
        def get_queries():
            return [q.strip() for q in custom_text.splitlines() if q.strip()]

    run_bench = st.button("Run benchmark")


# --------------------------- Benchmark output ---------------------------
if run_bench:
    queries = get_queries()
    if not bench_models or not queries:
        st.warning("Select at least one model and provide at least one query.")
    else:
        st.info("Running benchmark... (this may take a while depending on model sizes)")
        df = benchmark_models(models=bench_models, queries=queries, rag_enabled=rag_enabled)
        st.success("Benchmark complete")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="benchmark_results.csv",
            mime="text/csv",
        )

st.divider()

# --------------------------- Interactive panel ---------------------------
st.subheader("💡 Ask a Network Question")
query = st.text_input("Enter your question")
show_json = st.checkbox("Show raw JSON", value=False)
col1, col2 = st.columns([1, 3])
with col1:
    go = st.button("Run")
with col2:
    st.caption("Tip: ask for paths, degrees, hop count, min/max link distance, or equipment/QoT questions.")

if go and query.strip():
    # For interactive runs we reset state to avoid cross-query contamination
    _reset_state()

    out = run(query, mode=prompt_mode, rag_enabled=rag_enabled)

    if not out.get("ok"):
        st.error("Model output could not be parsed as JSON.")
        st.code(out.get("raw", ""))
    else:
        llm_json = out.get("llm_json", {})
        st.success("✅ Parsed JSON from model")

        # ---------------- Friendly answer rendering ----------------
        def _format_result(result):
            if result is None:
                return None
            if isinstance(result, (str, int, float)):
                return str(result)
            if not isinstance(result, dict):
                return str(result)

            # Prefer explicit summary if provided by the model
            summary = result.get("summary")
            if isinstance(summary, str) and summary.strip():
                return summary.strip()

            # Common keys (best effort)
            if "propagation_delay_s" in result:
                val = result.get("propagation_delay_s")
                try:
                    ms = float(val) * 1000.0
                    return f"Propagation delay: {ms:.3f} ms ({float(val):.6f} s)"
                except Exception:
                    return f"Propagation delay (s): {val}"
            if "path" in result and isinstance(result.get("path"), list):
                path = " → ".join([str(x) for x in result.get("path")])
                hop = result.get("hop_count")
                dist = result.get("total_distance_km")
                parts = [f"Path: {path}"]
                if hop is not None:
                    parts.append(f"Hop count: {hop}")
                if dist is not None:
                    parts.append(f"Total distance: {dist} km")
                return " | ".join(parts)

            # Equipment-style outputs
            if "nodes_to_deploy" in result and isinstance(result.get("nodes_to_deploy"), list):
                n = len(result.get("nodes_to_deploy"))
                return f"Equipment deployment list returned for {n} nodes. See tables or raw JSON for details."

            # Fallback: compact key: value lines
            lines = []
            for k, v in result.items():
                if isinstance(v, (dict, list)):
                    continue
                lines.append(f"{k}: {v}")
            return "\n".join(lines) if lines else None

        st.write("### Answer")
        task = llm_json.get("task") if isinstance(llm_json, dict) else None
        if isinstance(task, str) and task.strip():
            st.caption(f"Task: {task}")

        result_obj = llm_json.get("result") if isinstance(llm_json, dict) else None
        friendly = _format_result(result_obj)
        if friendly:
            st.write(friendly)
        else:
            st.write("(No concise answer produced; see tables/plot or enable raw JSON.)")

        if show_json:
            st.write("### Raw JSON")
            st.code(json.dumps(llm_json, indent=2), language="json")

        # Token usage
        st.caption(
            f"Prompt tokens: {out.get('prompt_tokens')} | Completion tokens: {out.get('completion_tokens')} | Total: {out.get('total_tokens')}"
        )

        # Plot
        if out.get("fig") is not None:
            st.write("### Plot")
            st.pyplot(out["fig"], clear_figure=False)

        # Tables
        node_df = out.get("node_df")
        link_df = out.get("link_df")
        qot_df = out.get("qot_df")

        if isinstance(node_df, pd.DataFrame):
            st.write("### Table: Nodes / Paths")
            st.dataframe(node_df, use_container_width=True)

        if isinstance(link_df, pd.DataFrame):
            st.write("### Table: Links")
            st.dataframe(link_df, use_container_width=True)

        if isinstance(qot_df, pd.DataFrame):
            st.write("### Table: QoT")
            st.dataframe(qot_df, use_container_width=True)

        # Free-form explanation
        expl = llm_json.get("explanation") if isinstance(llm_json, dict) else None
        if isinstance(expl, str) and expl.strip():
            st.write("### Explanation")
            st.write(expl)
