
"""llm_agent.py

NetSoft 2026 edition (LLM-first analytics + optional RAG)
--------------------------------------------------------

This agent intentionally does NOT ask the LLM to emit Python function calls.
Instead, it:
  1) Builds (or loads) a topology in Python (structure only)
  2) Sends a JSON-friendly snapshot of the graph + optional context
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

import ollama
import pandas as pd
import psutil
import yaml

from equipment_manager import EquipmentManager
from graph_processor import NetworkGraph
from qot_analyzer import QoTAnalyzer

# Fiber reference table (max values from the provided fiber table image).
# Used to provide deterministic attenuation/delay constants to the LLM.
try:
    from qot_utils import FIBER_REFERENCE  # type: ignore
except Exception:
    FIBER_REFERENCE = {}


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
    if mode == "none":
        return {**base, "temperature": 0.2, "num_predict": 600}
    if mode == "zero-shot":
        return {**base, "temperature": 0.0, "num_predict": 520}
    if mode == "one-shot":
        return {**base, "temperature": 0.0, "num_predict": 700}
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


# ------------------------------ JSON extraction ------------------------------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_first_json(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object from a model reply."""
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

    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return [_json_sanitize(x) for x in list(obj)]
        except Exception:
            pass

    try:
        import numpy as np

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except Exception:
        pass

    return obj


# ------------------------------ Query intent helpers ------------------------------

def _query_flags(q: str) -> Dict[str, bool]:
    ql = (q or "").strip().lower()

    scalar_hint = bool(re.search(r"\b(how many|total number of|total count|count of|number of)\b", ql))

    viz_any = bool(
        re.search(r"\b(plot|draw|visuali[sz]e|render|diagram|topology|graph|network)\b", ql)
        or bool(re.search(r"\b(from csv|csv files?)\b", ql))
    )

    viz_full = bool(
        re.search(r"\b(draw|plot|visuali[sz]e|render|create|show)\b.*\b(topology|graph|network)\b", ql)
        or bool(re.search(r"\b(create|draw|plot|visuali[sz]e|render)\b.*\b(from csv|csv files?)\b", ql))
    )

    viz_path = bool(re.search(r"\b(show|draw|plot|visuali[sz]e)\b.*\b(shortest path|path)\b", ql))

    random_gen = bool(re.search(r"\b(generate|create|make)\b.*\b(random)\b.*\b(topology|graph|network)\b", ql))

    # If we generate a topology, we should also plot it by default.
    if random_gen:
        viz_any = True
        viz_full = True

    return {
        "scalar_hint": scalar_hint,
        "viz_any": viz_any,
        "viz_full": viz_full,
        "viz_path": viz_path,
        "random_gen": random_gen,
    }


def _is_in_scope(q: str) -> bool:
    """Cheap Python-side domain gate to prevent unrelated questions reaching the LLM."""
    ql = (q or "").lower()

    network_terms = [
        "topology",
        "graph",
        "node",
        "link",
        "path",
        "shortest",
        "hop",
        "degree",
        "distance",
        "equipment",
        "roamd",
        "roadm",
        "amp",
        "amplifier",
        "pre-amp",
        "preamp",
        "booster",
        "ila",
        "attenuation",
        "propagation",
        "delay",
        "qot",
        "osnr",
        "gsnr",
        "snr",
        "fiber",
        "ssmf",
        "hcf",
        "csv",
        "random topology",
    ]

    oos_terms = [
        "holiday",
        "vacation",
        "travel",
        "berlin",
        "paris",
        "london",
        "make a cup of tea",
        "recipe",
        "cook",
        "hotel",
        "flight",
        "restaurant",
    ]

    has_network = any(t in ql for t in network_terms)
    has_oos = any(t in ql for t in oos_terms)

    if has_oos and not has_network:
        return False
    if not has_network:
        return False
    return True


# ------------------------------ Prompting ------------------------------
_ONE_SHOT_EXAMPLE = {
    "question": "Show the shortest path from M1 to M3 and plot it",
    "answer": {
        "result": {"path": ["M1", "M2", "M3"], "total_distance_km": 123.4, "hop_count": 2},
        "plot": {
            "title": "Shortest path M1 to M3",
            "highlight_edges": [["M1", "M2"], ["M2", "M3"]],
            "highlight_nodes": ["M1", "M2", "M3"],
        },
    },
}


def build_messages(query: str, mode: str, rag_enabled: bool) -> List[Dict[str, str]]:
    topology = _json_sanitize(net.topology_snapshot())

    flags = _query_flags(query)

    equip_hint = bool(
        re.search(
            r"\b(equipment|deploy|router|switch|amplifier|ila|qot|osnr|g?snr|ber|nf|gain|pre-amp|preamp|booster)\b",
            query,
            re.IGNORECASE,
        )
    )
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
                equip_tables_csv = "\n\n".join(parts)[:12000]
        except Exception:
            equip_tables_csv = None

    schema_rules = (
        "Return exactly ONE valid JSON object. No markdown, no code fences, no extra text.\n"
        "Allowed top-level keys ONLY: result, tables, plot.\n"
        "Do NOT include any other top-level keys.\n"
        "Do NOT place 'plot' inside 'result'. 'plot' MUST be top-level.\n"
        "\n"
        "DOMAIN SCOPE (HARD RULE):\n"
        "You may ONLY answer questions about: optical network topology (nodes/links), graph metrics (paths, degrees, hops), "
        "equipment deployment (from reference.equipment_tables_csv), and QoT (distance, attenuation, propagation delay).\n"
        "If the user request is outside this scope (travel, food, general knowledge, etc.), return ONLY:\n"
        '{ "result": { "error": "OUT_OF_SCOPE", "message": "Only optical network topology/equipment/QoT questions are supported." } }\n'
        "\n"
        "OUTPUT MINIMALITY (HARD RULE):\n"
        "- If the user asks for a single scalar value (count/total/how many), return ONLY:\n"
        '  { "result": { "value": <number> } }\n'
        "  Do NOT include plot or tables.\n"
        "- Do NOT add any commentary or summaries unless the user explicitly asks.\n"
        "\n"
        "RANDOM TOPOLOGY GENERATION (HARD RULE):\n"
        "If the user asks to generate/create a random topology/graph/network, you MUST request a Python-side topology change by returning:\n"
        '  { "result": { "topology_modifications": { "generate_random_topology": { ... } } }, "plot": { ... } }\n'
        "The generate_random_topology object may include: network_type/kind, num_nodes/num_metro, avg_degree, min_distance_km, max_distance_km, seed.\n"
        "When generating a random topology, also return a plot that draws the full topology (highlight_edges='ALL', highlight_nodes='ALL').\n"
        "\n"
        "PLOT CONTRACT (HARD RULE):\n"
        "If the user asks to show/draw/plot/visualize/render/create a topology or graph, you MUST return a top-level 'plot' object.\n"
        "The plot object MUST be small and MUST NOT contain 'data', 'nodes', 'edges', 'links', or any raw topology.\n"
        "The ONLY allowed keys inside plot are:\n"
        "  - title: string\n"
        "  - highlight_edges: 'ALL' or true or list of [u, v] pairs\n"
        "  - highlight_nodes: 'ALL' or true or list of node ids\n"
        "  - highlight_color: optional string\n"
        "\n"
        "Plot behavior rules:\n"
        "- For full-topology requests: set highlight_edges='ALL' and highlight_nodes='ALL'.\n"
        "- For path requests (e.g., shortest path M1 to M3) when visualization is requested:\n"
        "  * compute the path\n"
        "  * set highlight_nodes to the path nodes\n"
        "  * set highlight_edges to ONLY the consecutive edges along that path\n"
        "  * do NOT use 'ALL'.\n"
        "\n"
        "GRAPH RULES:\n"
        "You MUST compute graph analytics yourself from the provided topology snapshot.\n"
        "Do NOT ask for Python/networkx.\n"
        "\n"
        "EQUIPMENT RULES:\n"
        "If asked about equipment, do NOT invent. Use reference.equipment_tables_csv if present.\n"
        "Return tables ONLY if explicitly requested.\n"
        "\n"
        "PROPAGATION/ATTENUATION RULES:\n"
        "CONSTRAINED FIBER OPTIMIZATION (HARD RULE):\n"
        "If the user asks to minimize propagation delay using different fiber pairs with constraints "
        "(e.g., 'HCF restricted to only 1 span', 'exactly 4 spans from M1 to M6'), YOU must:\n"
        "1) Enumerate all simple paths between SRC and DST from the topology snapshot.\n"
        "2) Discard paths not matching the required span count (hop count).\n"
        "3) For each remaining path, assign fiber types per span such that constraints are met "
        "(e.g., at most one HCF span).\n"
        "4) Use reference.fiber_reference to compute per-span and total propagation delay.\n"
        "5) Select the combination with minimum total delay.\n"
        "6) Return JSON with: path, hop_count, fiber_assignment_per_span, total_propagation_delay_ms.\n"
        "Do NOT ask Python to optimize. Do NOT invent constants.\n"

        "If asked about propagation delay and/or attenuation, use reference.fiber_reference.\n"
        "Compute:\n"
        "  - total_distance_km = sum(edge.distance_km) along the chosen path\n"
        "  - propagation_delay_ms = total_distance_km * propagation_delay_ms_per_km\n"
        "  - attenuation_dB = total_distance_km * attenuation_dB_per_km\n"
    )

    # Expand fiber reference with ms/km for deterministic arithmetic.
    fiber_reference: Dict[str, Dict[str, float]] = {}
    try:
        for k, v in (FIBER_REFERENCE or {}).items():
            att = float(v.get("attenuation_dB_per_km"))
            delay_us = float(v.get("propagation_delay_us_per_km"))
            fiber_reference[str(k)] = {
                "attenuation_dB_per_km": att,
                "propagation_delay_us_per_km": delay_us,
                "propagation_delay_ms_per_km": delay_us / 1000.0,
            }
    except Exception:
        fiber_reference = {}

    user_payload: Dict[str, Any] = {
        "user_query": query,
        "topology": topology,
        "reference": {
            "ila_spacing_km": getattr(net, "ila_spacing_km", None),
            "qot_params": getattr(net, "qot_params", None),
            "equipment_tables_csv": equip_tables_csv,
            "fiber_reference": fiber_reference,
        },
        "notes": [
            "Distances are in km (edge.distance_km).",
            "Edges are undirected.",
            "link_id prefix ML/AL indicates link family.",
            "Fiber reference keys supported: ssmf, bend_insensitive_smf, hcf.",
            "If the user does not specify a fiber type, default to ssmf.",
        ],
    }

    # Visualization notes
    if flags["viz_full"]:
        user_payload["notes"].append(
            "Visualization requested: return top-level plot with highlight_edges='ALL' and highlight_nodes='ALL'."
        )
    if flags["viz_path"]:
        user_payload["notes"].append(
            "Visualization requested for a path: return top-level plot with highlight_nodes=path nodes and highlight_edges=only consecutive edges along that path (no ALL)."
        )
    if flags["random_gen"]:
        user_payload["notes"].append(
            "Random topology generation requested: include result.topology_modifications.generate_random_topology with parameters (network_type, num_nodes, avg_degree, min_distance_km, max_distance_km, seed)."
        )

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

    sys = schema_rules + "\nDouble-check your arithmetic."
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user_payload)},
    ]


# ------------------------------------ run() ------------------------------------
def run(query: str, mode: str = "zero-shot", rag_enabled: bool = False) -> Dict[str, Any]:
    """Run one interactive query."""
    global _TOPO_READY

    if not _TOPO_READY:
        net.build_topology(use_csv=True, visualize=False)
        _TOPO_READY = True

    if not _is_in_scope(query):
        oos = {
            "result": {
                "error": "OUT_OF_SCOPE",
                "message": "Only optical network topology/equipment/QoT questions are supported.",
            }
        }
        raw = json.dumps(oos)
        return {
            "ok": True,
            "llm_json": oos,
            "fig": None,
            "node_df": None,
            "link_df": None,
            "qot_df": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "raw": raw,
        }

    flags = _query_flags(query)

    messages = build_messages(query, mode, rag_enabled)
    raw, prompt_tokens, completion_tokens, total_tokens = call_llm(messages, mode=mode)

    chat_history.append({"role": "user", "content": query})
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
        llm_json = {"result": {"error": "PARSE_ERROR", "message": str(e), "raw": raw[:2000]}}
        ok = False

    # --- Post-parse sanitation for robustness ---
    if isinstance(llm_json, dict):
        res = llm_json.get("result")

        # 1) If model incorrectly nested plot under result, lift it to top-level.
        if isinstance(res, dict) and isinstance(res.get("plot"), dict) and not isinstance(llm_json.get("plot"), dict):
            llm_json["plot"] = res.pop("plot")

        # 2) If scalar query, strip plot/tables unless user explicitly requested visualization.
        if isinstance(res, dict) and "value" in res and flags.get("scalar_hint") and not flags.get("viz_any"):
            llm_json.pop("plot", None)
            llm_json.pop("tables", None)

        # 3) If plot contains forbidden heavy keys, drop them.
        plot = llm_json.get("plot")
        if isinstance(plot, dict):
            for forbidden in ["data", "nodes", "edges", "links"]:
                plot.pop(forbidden, None)

    # Apply requested topology edits BEFORE rendering plot.
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
                        "raw_output": out.get("raw"),
                    }
                )

    return pd.DataFrame(rows)


def run_experiment_from_yaml(models: List[str], yaml_file: str = "questions.yaml", rag_enabled: bool = False) -> pd.DataFrame:
    queries = load_benchmark_queries_from_yaml(yaml_file)
    return benchmark_models(models=models, queries=queries, rag_enabled=rag_enabled)

