"""llm_agent.py

NetSoft 2026 edition (LLM-first analytics + optional RAG)
--------------------------------------------------------

This agent intentionally does NOT ask the LLM to emit Python function calls.
Instead, it:
  1) Builds (or loads) a topology in Python (structure only)
  2) Sends a JSON-friendly snapshot of the graph + optional RAG context
  3) Asks the LLM to compute analytics (paths, degrees, hop count, etc.)
  4) Parses ONE strict JSON object from the LLM
  5) Optionally renders a plot using the LLM's "plot" spec

Important policy for this codebase:
  - Graph analytics are computed by the LLM.
  - QoT calculations remain in Python (reference model), as requested.

Expected files in project root:
  - distance.csv
  - city.csv
  - questions.yaml
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import psutil
import yaml
import ollama

from graph_processor import NetworkGraph
from qot_analyzer import QoTAnalyzer
from equipment_manager import EquipmentManager


# ------------------------------ Model registry ------------------------------
VALID_MODELS = {
    "qwen2.5:14b",
    "qwen2.5:7b",
    "llama3:8b",
    "llama3:latest",
    "mistral:7b",
    "gemma:7b",
    "gemma2:latest",
    "gemma3:4b",
}

_ACTIVE_MODEL = "llama3:8b"


def set_model(name: str) -> None:
    global _ACTIVE_MODEL
    if name not in VALID_MODELS:
        raise ValueError(f"Model '{name}' is not installed. Update VALID_MODELS or pull it via ollama.")
    _ACTIVE_MODEL = name


def get_model() -> str:
    return _ACTIVE_MODEL


# ------------------------------ Shared state ------------------------------
net = NetworkGraph("distance.csv", "city.csv")
qot = QoTAnalyzer(net)
eqp_mngr = EquipmentManager(net)

chat_history: List[Dict[str, Any]] = []
_TOPO_READY = False


def _reset_state() -> None:
    """Reset global state between runs/benchmarks for fair comparisons."""
    global chat_history, _TOPO_READY, net, qot, eqp_mngr
    chat_history.clear()
    _TOPO_READY = False
    net = NetworkGraph("distance.csv", "city.csv")
    qot = QoTAnalyzer(net)
    eqp_mngr = EquipmentManager(net)


# ------------------------- YAML query loader -------------------------
def _safe_fill(template: str, values: dict) -> str:
    def repl(m):
        key = m.group(1)
        return str(values.get(key, m.group(0)))

    return re.sub(r"\{([A-Za-z0-9_]+)\}", repl, template)


def load_benchmark_queries_from_yaml(
    yaml_file: str = "questions.yaml",
    default_vars: Optional[dict] = None,
) -> List[str]:
    default_vars = default_vars or {
        "NETWORK_TYPE": "metro",
        "NUM_METRO": 5,
        "NUM_ACCESS": 2,
        "AVG_DEGREE": 2,
        "MIN_DIST": 1,
        "MAX_DIST": 10,
        "SRC": "M1",
        "DST": "M3",
        "LINK_TYPE": "ML",
        "DISTANCE": 50,
    }
    if not os.path.exists(yaml_file):
        return []
    with open(yaml_file, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []

    queries: List[str] = []
    for item in data:
        qt = item.get("query_template")
        if isinstance(qt, str) and qt.strip():
            queries.append(_safe_fill(qt, default_vars))

    seen = set()
    dedup: List[str] = []
    for q in queries:
        if q not in seen:
            dedup.append(q)
            seen.add(q)
    return dedup


# ------------------------------ Ollama call ------------------------------
_DEF_SEED = 12345


def _decode_options_for_mode(mode: str) -> dict:
    base = {"seed": _DEF_SEED, "top_p": 1.0}
    # "none" means: no extra scaffolding other than schema rules
    if mode == "none":
        return {**base, "temperature": 0.2, "num_predict": 600}
    if mode == "zero-shot":
        return {**base, "temperature": 0.0, "num_predict": 520}
    if mode == "one-shot":
        return {**base, "temperature": 0.0, "num_predict": 700}
    # few-shot
    return {**base, "temperature": 0.05, "num_predict": 800}


def call_llm(messages: List[Dict[str, str]], mode: str = "zero-shot"):
    r = ollama.chat(model=_ACTIVE_MODEL, messages=messages, options=_decode_options_for_mode(mode))
    content = r["message"]["content"].strip()
    prompt_tokens = r.get("prompt_eval_count")
    completion_tokens = r.get("eval_count")
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return content, prompt_tokens, completion_tokens, total_tokens


# --------------------------- GPU / CPU logging ---------------------------
def read_gpu_usage():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        pynvml.nvmlShutdown()
        return power, util
    except Exception:
        return None, None


def read_cpu_usage():
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        return cpu_percent, mem
    except Exception:
        return None, None


# ------------------------------ RAG (lightweight) ------------------------------
def _load_rag_context(rag_enabled: bool) -> str:
    """Simple RAG: return a context string.

    We reuse equipment + QoT reference tables and any optional user notes in
    ./rag_corpus/.
    """
    if not rag_enabled:
        return ""

    parts: List[str] = []
    try:
        eqp_mngr.deploy_equipment()
        summary = eqp_mngr.summarize_equipment(visualize=False)
        node_df = summary.get("node_df", pd.DataFrame())
        link_df = summary.get("link_df", pd.DataFrame())
        if isinstance(node_df, pd.DataFrame) and not node_df.empty:
            parts.append("EQUIPMENT_NODE_TABLE\n" + node_df.to_csv(index=False))
        if isinstance(link_df, pd.DataFrame) and not link_df.empty:
            parts.append("EQUIPMENT_LINK_TABLE\n" + link_df.to_csv(index=False))
    except Exception:
        pass

    corpus_dir = "rag_corpus"
    if os.path.isdir(corpus_dir):
        for fn in sorted(os.listdir(corpus_dir)):
            if not fn.lower().endswith((".txt", ".md", ".csv")):
                continue
            fp = os.path.join(corpus_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    parts.append(f"FILE:{fn}\n" + f.read())
            except Exception:
                continue

    return "\n\n".join(parts)[:20000]


# ------------------------------ JSON extraction ------------------------------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_first_json(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object from a model reply."""
    # Many models wrap JSON in fenced blocks (```json ... ```). We must NOT
    # delete the contents (older regex approaches did), only remove the fence
    # markers so the JSON remains extractable.
    cleaned = text.strip()
    cleaned = re.sub(r"```\s*(json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()
    m = _JSON_RE.search(cleaned)
    if not m:
        raise ValueError("No JSON object found in model response")
    candidate = m.group(0)
    return json.loads(candidate)


def _json_sanitize(obj: Any) -> Any:
    """Recursively convert non-JSON-safe objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, set):
        return [_json_sanitize(x) for x in sorted(obj, key=lambda x: str(x))]

    # NetworkX views, dict views, pandas Index, etc.
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return [_json_sanitize(x) for x in list(obj)]
        except Exception:
            pass

    # numpy scalars
    try:
        import numpy as np

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except Exception:
        pass

    return obj


# ------------------------------ Prompting ------------------------------
_ONE_SHOT_EXAMPLE = {
    "question": "Find the shortest path from M1 to M3",
    "answer": {
        "task": "shortest_path",
        "result": {"path": ["M1", "M2", "M3"], "total_distance_km": 123.4, "hop_count": 2},
        "explanation": "Computed by enumerating simple paths and selecting minimum total distance.",
        "plot": {"title": "Shortest path M1 to M3", "highlight_edges": [["M1", "M2"], ["M2", "M3"]]},
    },
}


def build_messages(query: str, mode: str, rag_enabled: bool) -> List[Dict[str, str]]:
    topology = _json_sanitize(net.topology_snapshot())
    rag = _load_rag_context(rag_enabled)

    # If the user is asking about equipment/QoT, include authoritative equipment tables
    # (computed by the hardcoded Python logic) to reduce hallucinations.
    equip_hint = bool(re.search(r"\b(equipment|deploy|router|switch|amplifier|ila|qot|osnr|g?snr|ber|nf|gain)\b", query, re.IGNORECASE))
    equip_tables_csv = None
    if equip_hint:
        try:
            eqp_mngr.deploy_equipment()
            s = eqp_mngr.summarize_equipment(visualize=False)
            node_df = s.get("node_df", pd.DataFrame())
            link_df = s.get("link_df", pd.DataFrame())
            parts = []
            if isinstance(node_df, pd.DataFrame) and not node_df.empty:
                parts.append("NODE_EQUIPMENT_TABLE_CSV\n" + node_df.to_csv(index=False))
            if isinstance(link_df, pd.DataFrame) and not link_df.empty:
                parts.append("LINK_EQUIPMENT_TABLE_CSV\n" + link_df.to_csv(index=False))
            if parts:
                equip_tables_csv = "\n\n".join(parts)
                equip_tables_csv = equip_tables_csv[:12000]
        except Exception:
            equip_tables_csv = None

    schema_rules = (
        "Return exactly ONE JSON object (no markdown, no code fences).\n"
        "JSON keys allowed: task, result, tables, explanation, plot, warnings.\n"
        "- task: short string describing what you did\n"
        "- result: object with your computed numeric/structured answer. Include a human-readable 'summary' string when possible.\n"
        "- tables: optional list of {name, rows} where rows is a list of objects\n"
        "- plot: optional object with: title, highlight_edges, highlight_nodes, highlight_color\n"
        "  * highlight_edges: either a list of [u,v] pairs OR the string 'ALL' to draw all edges\n"
        "  * highlight_nodes: either a list of node ids OR the string 'ALL' to draw all nodes\n"
        "  * If the user asks to plot/visualize the topology, prefer highlight_edges='ALL'\n"
        "\n"
        "You MUST compute graph analytics yourself from the provided topology snapshot.\n"
        "Do NOT ask Python/networkx to compute shortest paths, degrees, etc.\n"
        "If asked about equipment deployment, do NOT invent. Use reference.equipment_tables_csv if present and return it as tables named 'node_equipment' and 'link_equipment'.\n"
    )

    user_payload: Dict[str, Any] = {
        "user_query": query,
        "topology": topology,
        "reference": {
            "ila_spacing_km": getattr(net, "ila_spacing_km", None),
            "qot_params": getattr(net, "qot_params", None),
            "equipment_tables_csv": equip_tables_csv,
        },
        "notes": [
            "Distances are in km (edge.distance_km).",
            "Edges are undirected.",
            "link_id prefix ML/AL indicates link family.",
        ],
    }

    if rag:
        user_payload["rag_context"] = rag

    if mode == "none":
        return [
            {"role": "system", "content": schema_rules},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

    if mode == "zero-shot":
        sys = schema_rules + "\nBe concise and deterministic."
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

    if mode == "one-shot":
        sys = schema_rules + "\nFollow the example format."
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps({"user_query": _ONE_SHOT_EXAMPLE["question"], "topology": topology})},
            {"role": "assistant", "content": json.dumps(_ONE_SHOT_EXAMPLE["answer"])},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

    # few-shot
    sys = schema_rules + "\nDouble-check your arithmetic."
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user_payload)},
    ]


# ------------------------------------ run() ------------------------------------
def run(query: str, mode: str = "zero-shot", rag_enabled: bool = False) -> Dict[str, Any]:
    """Run one interactive query.

    Returns a dict:
      - ok: bool
      - llm_json: parsed JSON (or parse_error JSON)
      - fig: matplotlib fig (optional)
      - node_df/link_df/qot_df: dataframes (optional)
      - token usage fields
    """
    global _TOPO_READY

    # Ensure topology ready
    if not _TOPO_READY:
        net.build_topology(use_csv=True, visualize=False)
        _TOPO_READY = True

    messages = build_messages(query, mode, rag_enabled)
    raw, prompt_tokens, completion_tokens, total_tokens = call_llm(messages, mode=mode)

    chat_history.append(
        {
            "role": "user",
            "content": query,
        }
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": raw,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    )

    try:
        llm_json = _extract_first_json(raw)
        ok = True
    except Exception as e:
        llm_json = {"task": "parse_error", "warnings": [str(e)], "result": {"raw": raw[:2000]}}
        ok = False

    # Optional: apply requested topology edits (LLM can include result.graph_edits)
    try:
        if isinstance(llm_json, dict) and isinstance(llm_json.get("result"), dict):
            edits = llm_json["result"].get("graph_edits") or llm_json["result"].get("topology_modifications")
            if isinstance(edits, dict):
                net.apply_graph_edits(edits)
    except Exception:
        pass

    # Optional plot
    fig = None
    plot = llm_json.get("plot") if isinstance(llm_json, dict) else None
    if isinstance(plot, dict):
        try:
            _, fig = net.render_plot(plot)
        except Exception:
            fig = None

    # Tables (optional)
    node_df = link_df = qot_df = None
    tables = llm_json.get("tables") if isinstance(llm_json, dict) else None
    if isinstance(tables, list):
        for t in tables:
            if not isinstance(t, dict):
                continue
            name = str(t.get("name") or "").lower()
            rows = t.get("rows")
            if not isinstance(rows, list):
                continue
            df = pd.DataFrame(rows)
            if any(k in name for k in ["node", "path"]):
                node_df = df
            elif "link" in name or "edge" in name:
                link_df = df
            elif "qot" in name:
                qot_df = df

    return {
        "ok": ok,
        "llm_json": llm_json,
        "fig": fig,
        "node_df": node_df,
        "link_df": link_df,
        "qot_df": qot_df,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw": raw,
    }


# ------------------------------ Benchmarking ------------------------------
def benchmark_models(models: List[str], queries: List[str], rag_enabled: bool = False) -> pd.DataFrame:
    rows: List[dict] = []
    for m in models:
        set_model(m)
        for q in queries:
            for mode in ["none", "zero-shot", "one-shot", "few-shot"]:
                _reset_state()
                t0 = time.time()
                gpu_before, _ = read_gpu_usage()
                out = run(q, mode=mode, rag_enabled=rag_enabled)
                latency = (time.time() - t0) * 1000.0
                gpu_after, gpu_util = read_gpu_usage()
                cpu, mem = read_cpu_usage()

                success = int(bool(out.get("ok")))

                rows.append(
                    {
                        "model": m,
                        "mode": mode,
                        "rag": int(bool(rag_enabled)),
                        "query": q,
                        "success": success,
                        "latency_ms": round(latency, 2),
                        "prompt_tokens": out.get("prompt_tokens"),
                        "completion_tokens": out.get("completion_tokens"),
                        "total_tokens": out.get("total_tokens"),
                        "gpu_power_w_before": gpu_before,
                        "gpu_power_w_after": gpu_after,
                        "gpu_util_percent": gpu_util,
                        "cpu_percent": cpu,
                        "mem_percent": mem,
                        "json_task": out.get("llm_json", {}).get("task") if isinstance(out.get("llm_json"), dict) else None,
                        "raw_output": out.get("raw"),
                    }
                )

    return pd.DataFrame(rows)


def run_experiment_from_yaml(models: List[str], yaml_file: str = "questions.yaml", rag_enabled: bool = False) -> pd.DataFrame:
    queries = load_benchmark_queries_from_yaml(yaml_file)
    return benchmark_models(models=models, queries=queries, rag_enabled=rag_enabled)
