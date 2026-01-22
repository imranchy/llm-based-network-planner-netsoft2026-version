#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import llm_agent


# ----------------------- CONFIG -----------------------
MODELS = [
    "qwen2.5:14b-instruct",
    "llama3.1:8b",
    "mistral-nemo:12b",
    "gemma2:9b",
]

QUERIES: List[str] = [
    "Q1: Create a topology from CSV files and visualize the full network topology.",
    "Q2: Create a topology from CSV files (no visualization).",
    "Q3: Draw the network topology from the CSV files and label each link with its distance.",

    "Q4: Show the shortest path from M4 to M7 and plot that path only (not the full topology).",
    "Q5: Show the shortest path from M2 to M7 and plot that path only (not the full topology).",
    "Q6: Show the shortest path from M1 to M4 and plot that path only (not the full topology).",
    "Q7: Show the longest path from M5 to M7 (by total distance) and return hop count and total distance.",
    "Q8: Show the longest path from M3 to M4 (by total distance) and return hop count and total distance.",
    "Q9: List all paths between M1 and M2 (no plot) and return how many such paths exist.",
    "Q10: Among all paths between M5 and M7, which path has the fewest hops? Return the path and hop count.",

    "Q11: Which node has the highest degree in the current topology? Return the node id and degree.",
    "Q12: Which node has the lowest degree? Return the node id and degree.",
    "Q13: Compute the average node degree, the minimum degree, and the maximum degree.",
    "Q14: Count how many nodes have degree 1.",
    "Q15: Count how many nodes have degree >= 3.",
    "Q16: Is there any isolated node (degree 0)? Return true/false.",
    "Q17: Find the top 3 nodes by degree and return them in descending order.",
    "Q19: Compute the average shortest-path length of the topology (in hops).",

    "Q25: Show ML links under 5 km (return a table, no plot).",
    "Q26: Show AL links under 10 km (return a table, no plot).",
    "Q27: Calculate average ML link distance (km).",
    "Q28: Calculate average AL link distance (km).",
    "Q29: Find maximum ML link distance and plot only that edge highlighted.",
    "Q30: Return the 5 longest links in the topology (any type) with endpoints, link_id, and distance_km.",

    "Q31: Deploy network equipment and then show a summary table for node equipment (no plot).",
    "Q32: After deploying equipment, how many ROADMs were placed in total? Return a scalar count.",
    "Q33: After deploying equipment, how many TRx elements were placed in total? Return a scalar count.",
    "Q34: After deploying equipment, count the total number of ILAs across all links.",
    "Q35: After deploying equipment, list all links that have an ILA deployed (link endpoints + equipment list).",
    "Q36: After deploying equipment, list all links that have a PreAmp deployed (link endpoints + equipment list).",
    "Q37: After deploying equipment, list all links that have an Amp deployed (link endpoints + equipment list).",
]


# ----------------------- HELPERS -----------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def model_dir_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").strip()


def query_id(query: str) -> str:
    m = re.match(r"^(Q\d{1,3})\b", (query or "").strip(), flags=re.IGNORECASE)
    if not m:
        return "QXXX"
    q = m.group(1).upper()
    num = int(re.sub(r"\D", "", q))
    return f"Q{num:03d}"


def proxy_accuracy_and_hallucination_from_llm_json(query: str, llm_json: Any) -> Tuple[int, int]:
    """
    PROXY ONLY (not your final paper metric):
      - proxy_accuracy = 1 if expected fields exist
      - proxy_hallucination = 1 if JSON exists but expected fields missing
    Honest error => neither accurate nor hallucination.
    """
    if not isinstance(llm_json, dict):
        return 0, 0

    payload = llm_json.get("result", llm_json)
    if isinstance(payload, dict) and "error" in payload:
        return 0, 0

    ql = (query or "").lower()
    expected: List[str] = []
    if "shortest path" in ql or "longest" in ql or "simple paths" in ql or "fewest hops" in ql:
        expected = ["path", "paths", "hop_count", "total_distance_km", "value"]
    elif any(k in ql for k in ["degree", "diameter", "average shortest-path", "clustering", "triangles", "bridges", "articulation", "bipartite"]):
        expected = ["value", "degree", "min_degree", "max_degree", "avg_degree", "diameter_hops"]
    elif any(k in ql for k in ["links under", "longest links", "maximum ml link", "average ml", "average al"]):
        expected = ["tables", "value"]
    elif any(k in ql for k in ["deploy", "roadm", "trx", "ila", "preamp", "amp"]):
        expected = ["tables", "value"]
    elif "topology" in ql or "csv" in ql or "visualize" in ql or "draw" in ql or "graph" in ql:
        expected = ["plot", "value", "tables"]

    ok = 1 if any(k in llm_json for k in expected) or (isinstance(payload, dict) and any(k in payload for k in expected)) else 0
    hall = 1 if ok == 0 else 0
    return ok, hall


def save_artifacts(artifacts_dir: str, model: str, query: str, out: Dict[str, Any]) -> None:
    mdir = os.path.join(artifacts_dir, model_dir_name(model))
    qdir = os.path.join(mdir, query_id(query))
    ensure_dir(qdir)

    # raw output (if present)
    raw_path = os.path.join(qdir, "raw_output.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(out.get("raw", "") or "")

    # structured json
    if isinstance(out.get("llm_json"), dict):
        out["llm_json"].pop("raw_llm_output", None)  # keep artifacts smaller if you store it inside llm_json
        llm_json_path = os.path.join(qdir, "llm_json.json")
        with open(llm_json_path, "w", encoding="utf-8") as f:
            import json
            json.dump(out["llm_json"], f, indent=2, ensure_ascii=False)

    # plot
    fig = out.get("fig")
    if fig is not None:
        try:
            plot_path = os.path.join(qdir, "plot.png")
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
        except Exception:
            pass

    # tables
    if isinstance(out.get("node_df"), pd.DataFrame) and not out["node_df"].empty:
        out["node_df"].to_csv(os.path.join(qdir, "node_df.csv"), index=False)
    if isinstance(out.get("link_df"), pd.DataFrame) and not out["link_df"].empty:
        out["link_df"].to_csv(os.path.join(qdir, "link_df.csv"), index=False)
    if isinstance(out.get("qot_df"), pd.DataFrame) and not out["qot_df"].empty:
        out["qot_df"].to_csv(os.path.join(qdir, "qot_df.csv"), index=False)


# ----------------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_out", default="summary_results.csv")
    p.add_argument("--artifacts_dir", default="artifacts")
    p.add_argument("--mode", default="none")
    p.add_argument("--no_reset", action="store_true", help="Speed mode: do not reset between queries.")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.artifacts_dir)

    rows: List[Dict[str, Any]] = []
    t0_bench = time.time()

    for model in MODELS:
        llm_agent.set_model(model)

        if args.no_reset:
            llm_agent._reset_state()

        for query in QUERIES:
            if not args.no_reset:
                llm_agent._reset_state()

            gpu_before_w, _ = llm_agent.read_gpu_usage()
            t0 = time.time()
            out = llm_agent.run(query, mode=args.mode, rag_enabled=False)
            latency_ms = (time.time() - t0) * 1000.0
            gpu_after_w, gpu_util = llm_agent.read_gpu_usage()
            cpu_percent, mem_percent = llm_agent.read_cpu_usage()

            success = int(bool(out.get("ok")))
            llm_json = out.get("llm_json")

            proxy_acc, proxy_hall = (0, 0)
            if success == 1:
                proxy_acc, proxy_hall = proxy_accuracy_and_hallucination_from_llm_json(query, llm_json)

            save_artifacts(args.artifacts_dir, model, query, out)

            rows.append(
                {
                    "model": model,
                    "mode": args.mode,
                    "query": query,
                    "success": success,
                    "proxy_accuracy": proxy_acc,
                    "proxy_hallucination": proxy_hall,
                    "latency_ms": float(latency_ms),
                    "prompt_tokens": out.get("prompt_tokens"),
                    "completion_tokens": out.get("completion_tokens"),
                    "total_tokens": out.get("total_tokens"),
                    "cpu_percent": cpu_percent,
                    "mem_percent": mem_percent,
                    "gpu_util_percent": gpu_util,
                    "gpu_power_w_before": gpu_before_w,
                    "gpu_power_w_after": gpu_after_w,
                }
            )

        print(f"Finished model: {model}")

    df = pd.DataFrame(rows)

    for c in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    summary = (
        df.groupby(["model", "mode"], as_index=False)
        .agg(
            n_queries=("query", "count"),
            success_rate_pct=("success", lambda s: round(100.0 * float(pd.Series(s).mean()), 2)),
            proxy_accuracy_pct=("proxy_accuracy", lambda s: round(100.0 * float(pd.Series(s).mean()), 2)),
            proxy_hallucination_pct=("proxy_hallucination", lambda s: round(100.0 * float(pd.Series(s).mean()), 2)),
            mean_latency_ms=("latency_ms", lambda s: round(float(pd.Series(s).mean()), 2)),
            avg_cpu_percent=("cpu_percent", lambda s: round(float(pd.Series(s).dropna().mean()), 2) if pd.Series(s).dropna().size else None),
            avg_mem_percent=("mem_percent", lambda s: round(float(pd.Series(s).dropna().mean()), 2) if pd.Series(s).dropna().size else None),
            avg_gpu_util_percent=("gpu_util_percent", lambda s: round(float(pd.Series(s).dropna().mean()), 2) if pd.Series(s).dropna().size else None),
            mean_prompt_tokens=("prompt_tokens", lambda s: round(float(pd.Series(s).dropna().mean()), 2) if pd.Series(s).dropna().size else None),
            mean_completion_tokens=("completion_tokens", lambda s: round(float(pd.Series(s).dropna().mean()), 2) if pd.Series(s).dropna().size else None),
            mean_total_tokens=("total_tokens", lambda s: round(float(pd.Series(s).dropna().mean()), 2) if pd.Series(s).dropna().size else None),
        )
    )

    summary.to_csv(args.summary_out, index=False)
    print(f"\nWrote summary to: {os.path.abspath(args.summary_out)}")
    print(f"Artifacts saved under: {os.path.abspath(args.artifacts_dir)}")
    print(f"Done in {time.time() - t0_bench:.2f}s")


if __name__ == "__main__":
    import sys
    sys.argv = [
        "benchmark_summary_only.py",
        "--summary_out", "summary_results.csv",
        "--artifacts_dir", "artifacts",
        "--mode", "none",
        # "--no_reset",
    ]
    main()
