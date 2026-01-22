"""llm_agent.py

NetSoft 2026 edition — Hybrid LLM + Python analytics
---------------------------------------------------

- Python performs deterministic computations:
  * topology build (from CSV)
  * shortest path / longest simple path / all simple paths count
  * fewest-hop path
  * graph metrics (degrees, diameter, average shortest path length)
  * link statistics (filters, averages, top-k)
  * equipment deployment + summary tables

- The LLM is a fallback for anything not covered.

Expected files in project root:
  - distance.csv
  - city.csv
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import ollama
import pandas as pd
import psutil

from equipment_manager import EquipmentManager
from graph_processor import NetworkGraph

# ------------------------------ Model registry ------------------------------
VALID_MODELS = {
    "qwen2.5:14b-instruct",
    "llama3.1:8b",
    "mistral-nemo:12b",
    "gemma2:9b",
}

_ACTIVE_MODEL = "qwen2.5:14b-instruct"


def set_model(name: str) -> None:
    global _ACTIVE_MODEL
    if name not in VALID_MODELS:
        raise ValueError(
            f"Model '{name}' is not installed. Update VALID_MODELS or pull it via ollama."
        )
    _ACTIVE_MODEL = name


def get_model() -> str:
    return _ACTIVE_MODEL


# ------------------------------ Shared state ------------------------------
net = NetworkGraph("distance.csv", "city.csv")
eqp_mngr = EquipmentManager(net)
_TOPO_READY = False


def _reset_state() -> None:
    """Reset global state between runs/benchmarks for fair comparisons."""
    global _TOPO_READY, net, eqp_mngr
    _TOPO_READY = False
    net = NetworkGraph("distance.csv", "city.csv")
    eqp_mngr = EquipmentManager(net)


# --------------------------- CPU / RAM / GPU logging ---------------------------
_NVML = {"ok": False, "h": None}


def _nvml_init_once() -> None:
    if _NVML["ok"]:
        return
    try:
        import pynvml

        pynvml.nvmlInit()
        _NVML["h"] = pynvml.nvmlDeviceGetHandleByIndex(0)
        _NVML["ok"] = True
    except Exception:
        _NVML["ok"] = False
        _NVML["h"] = None


def read_gpu_usage() -> Tuple[Optional[float], Optional[float]]:
    try:
        import pynvml

        _nvml_init_once()
        if not _NVML["ok"] or _NVML["h"] is None:
            return None, None
        h = _NVML["h"]
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        util = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        return float(power), float(util)
    except Exception:
        return None, None


def read_cpu_usage() -> Tuple[Optional[float], Optional[float]]:
    try:
        cpu_percent = float(psutil.cpu_percent(interval=None))
        mem_percent = float(psutil.virtual_memory().percent)
        return cpu_percent, mem_percent
    except Exception:
        return None, None


# ------------------------------ Ollama call ------------------------------
_DEF_SEED = 12345


def _decode_options() -> dict:
    # Deterministic benchmarking-friendly options
    return {"seed": _DEF_SEED, "top_p": 1.0, "temperature": 0.0, "num_predict": 500}


def call_llm(messages: List[Dict[str, str]]) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    r = ollama.chat(model=_ACTIVE_MODEL, messages=messages, options=_decode_options())
    content = (r["message"]["content"] or "").strip()
    prompt_tokens = r.get("prompt_eval_count")
    completion_tokens = r.get("eval_count")
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return content, prompt_tokens, completion_tokens, total_tokens


# ------------------------------ JSON extraction ------------------------------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_first_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"```\s*(json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    m = _JSON_RE.search(cleaned)
    if not m:
        raise ValueError("No JSON object found in model response")
    return json.loads(m.group(0))


# ------------------------------ Query flags / parsing ------------------------------
_NODE_RE = re.compile(r"\b(M\d+)\b", re.IGNORECASE)


def _query_flags(q: str) -> Dict[str, bool]:
    ql = (q or "").lower()

    no_plot = bool(re.search(r"\b(no plot|no visualization|without visualization)\b", ql))
    viz_any = bool(re.search(r"\b(plot|draw|visuali[sz]e|render)\b", ql)) and not no_plot

    path_only = bool(re.search(r"\b(path only|only the path|not the full topology|not the topology|do not show full topology)\b", ql))

    # IMPORTANT: include "longest path" (not only "longest simple path")
    path_intent = bool(
        re.search(
            r"\b(shortest path|longest simple path|longest path|list all simple paths|fewest hops)\b",
            ql,
        )
    )

    return {
        "no_plot": no_plot,
        "viz_any": viz_any,
        "path_only": path_only,
        "equip": bool(re.search(r"\b(deploy|equipment|roadm|trx|ila|preamp|amp)\b", ql)),
        "path": path_intent,
        "topology": bool(re.search(r"\b(topology|from csv|csv files?)\b", ql)),
        "link_stats": bool(
            re.search(
                r"\b(ml links|al links|longest links|average .* link distance|max .* link distance|maximum .* link distance)\b",
                ql,
            )
        ),
        "graph_stats": bool(
            re.search(
                r"\b(highest degree|lowest degree|average node degree|degree 1|degree >=|isolated|top 3|diameter|average shortest-path length)\b",
                ql,
            )
        ),
    }


def extract_src_dst(query: str) -> Tuple[Optional[str], Optional[str]]:
    q = (query or "").strip()
    m = re.search(r"\bfrom\s+(M\d+)\s+to\s+(M\d+)\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).upper()
    m = re.search(r"\bbetween\s+(M\d+)\s+and\s+(M\d+)\b", q, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).upper()
    nodes = [x.upper() for x in _NODE_RE.findall(q)]
    if len(nodes) >= 2:
        return nodes[0], nodes[1]
    return None, None


def _path_edges(path: Any) -> List[List[str]]:
    if not isinstance(path, list) or len(path) < 2:
        return []
    out: List[List[str]] = []
    for i in range(len(path) - 1):
        out.append([str(path[i]), str(path[i + 1])])
    return out


def _edges_df() -> pd.DataFrame:
    snap = net.topology_snapshot()
    edges = snap.get("edges", []) if isinstance(snap, dict) else []
    df = pd.DataFrame(edges)

    # Normalize common column names
    if "u" not in df.columns and "source" in df.columns:
        df["u"] = df["source"]
    if "v" not in df.columns and "target" in df.columns:
        df["v"] = df["target"]

    for c in ["u", "v", "link_id", "type", "distance_km"]:
        if c not in df.columns:
            df[c] = None
    return df


# ------------------------------ Deterministic handlers ------------------------------
def _answer_topology(flags: Dict[str, bool]) -> Tuple[Dict[str, Any], Any]:
    """
    Return JSON + optional fig.
    """
    llm_json: Dict[str, Any] = {"result": {"value": "TOPOLOGY_READY"}}
    fig = None
    if flags["viz_any"]:
        plot_spec = {"title": "Network topology", "highlight_edges": "ALL", "highlight_nodes": "ALL"}
        try:
            _, fig = net.render_plot(plot_spec)
        except Exception:
            fig = None
        llm_json["plot"] = plot_spec
    return llm_json, fig


def _answer_paths(query: str, flags: Dict[str, bool]) -> Tuple[Dict[str, Any], Any, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic path answering.

    Key fix vs your previous version:
      - NEVER use highlight_edges="AUTO"
      - For path-only plots, we explicitly build highlight_edges from the chosen path.
      - We do NOT rely on query_network_path(..., visualize=True) to avoid full-topology plots.
    """
    ql = (query or "").lower()
    src, dst = extract_src_dst(query)
    if not src or not dst:
        return (
            {"result": {"error": "NO_PATH", "message": "Could not extract SRC/DST from query."}},
            None,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # Helper: get best path row from query_network_path without plotting
    def _best_path(mode: str) -> Tuple[Optional[List[str]], Optional[float], Optional[int], pd.DataFrame, pd.DataFrame]:
        fig0, node_df0, link_df0 = net.query_network_path(src, dst, mode=mode, visualize=False)
        if not isinstance(node_df0, pd.DataFrame) or node_df0.empty:
            return None, None, None, pd.DataFrame(), pd.DataFrame()
        row = node_df0.iloc[0].to_dict()
        path = row.get("Nodes")
        total_km = row.get("Total Distance (km)")
        hops = row.get("Hop Count")

        # Normalize
        if isinstance(path, str):
            # handle "A → B → C" style
            parts = [p.strip() for p in path.replace("->", "→").split("→") if p.strip()]
            path = parts
        if isinstance(path, tuple):
            path = list(path)

        try:
            hops_i = int(hops) if pd.notna(hops) else None
        except Exception:
            hops_i = None

        try:
            total_f = float(total_km) if pd.notna(total_km) else None
        except Exception:
            total_f = None

        return path, total_f, hops_i, node_df0, (link_df0 if isinstance(link_df0, pd.DataFrame) else pd.DataFrame())

    # SHORTTEST
    if "shortest path" in ql:
        path, total_km, hops, node_df, link_df = _best_path("shortest")
        if not path:
            return {"result": {"error": "NO_PATH", "message": "No path returned by Python."}}, None, pd.DataFrame(), pd.DataFrame()

        llm_json: Dict[str, Any] = {
            "result": {"path": path, "hop_count": hops, "total_distance_km": total_km}
        }

        fig = None
        if flags["viz_any"]:
            edges = _path_edges(path)
            llm_json["plot"] = {
                "title": f"Shortest Path: {src} → {dst}",
                "highlight_nodes": path,
                "highlight_edges": edges,
                "only_highlight": bool(flags.get("path_only", False)),
            }
            # If user asked for path-only, render only highlighted subgraph; otherwise use the path helper
            try:
                if bool(flags.get("path_only", False)):
                    _, fig = net.render_plot(llm_json["plot"])
                else:
                    _, fig = net._visualize_path(path, title=f"Shortest Path: {src} → {dst}", highlight_color="green")
            except Exception:
                try:
                    _, fig = net.render_plot(llm_json["plot"])
                except Exception:
                    fig = None

        return llm_json, fig, node_df, link_df

    # LONGEST (accept both "longest simple path" and "longest path")
    if ("longest simple path" in ql) or ("longest path" in ql):
        path, total_km, hops, node_df, link_df = _best_path("longest")
        if not path:
            return {"result": {"error": "NO_PATH", "message": "No path returned by Python."}}, None, pd.DataFrame(), pd.DataFrame()

        llm_json = {
            "result": {"path": path, "hop_count": hops, "total_distance_km": total_km}
        }

        fig = None
        if flags["viz_any"]:
            edges = _path_edges(path)
            llm_json["plot"] = {
                "title": f"Longest Path: {src} → {dst}",
                "highlight_nodes": path,
                "highlight_edges": edges,
                "only_highlight": bool(flags.get("path_only", False)),
            }
            try:
                if bool(flags.get("path_only", False)):
                    _, fig = net.render_plot(llm_json["plot"])
                else:
                    _, fig = net._visualize_path(path, title=f"Longest Path: {src} → {dst}", highlight_color="red")
            except Exception:
                try:
                    _, fig = net.render_plot(llm_json["plot"])
                except Exception:
                    fig = None

        return llm_json, fig, node_df, link_df

    # COUNT ALL SIMPLE PATHS
    if "list all simple paths" in ql:
        fig0, node_df, link_df = net.query_network_path(src, dst, mode="all", visualize=False)
        n = int(len(node_df)) if isinstance(node_df, pd.DataFrame) else 0
        return {"result": {"value": n}}, None, (node_df if isinstance(node_df, pd.DataFrame) else pd.DataFrame()), (link_df if isinstance(link_df, pd.DataFrame) else pd.DataFrame())

    # FEWEST HOPS
    if "fewest hops" in ql:
        fig0, node_df, link_df = net.query_network_path(src, dst, mode="all", visualize=False)
        if not isinstance(node_df, pd.DataFrame) or node_df.empty:
            return {"result": {"error": "NO_PATH", "message": "No paths exist."}}, None, pd.DataFrame(), pd.DataFrame()

        # Pick path with minimum hop count (tie-breaker: lower distance)
        tmp = node_df.copy()
        tmp["Hop Count"] = pd.to_numeric(tmp["Hop Count"], errors="coerce")
        tmp["Total Distance (km)"] = pd.to_numeric(tmp["Total Distance (km)"], errors="coerce")
        tmp = tmp.dropna(subset=["Hop Count"])
        tmp = tmp.sort_values(["Hop Count", "Total Distance (km)"], ascending=[True, True])

        best = tmp.iloc[0].to_dict()
        path = best.get("Nodes")
        if isinstance(path, str):
            parts = [p.strip() for p in path.replace("->", "→").split("→") if p.strip()]
            path = parts
        if isinstance(path, tuple):
            path = list(path)

        hops = int(best.get("Hop Count")) if pd.notna(best.get("Hop Count")) else None

        llm_json = {"result": {"path": path, "hop_count": hops}}

        fig = None
        if flags["viz_any"] and isinstance(path, list) and len(path) >= 2:
            edges = _path_edges(path)
            llm_json["plot"] = {
                "title": f"Fewest hops: {src} → {dst}",
                "highlight_nodes": path,
                "highlight_edges": edges,
                "only_highlight": bool(flags.get("path_only", False)),
            }
            try:
                if bool(flags.get("path_only", False)):
                    _, fig = net.render_plot(llm_json["plot"])
                else:
                    _, fig = net._visualize_path(path, title=f"Fewest hops: {src} → {dst}", highlight_color="blue")
            except Exception:
                try:
                    _, fig = net.render_plot(llm_json["plot"])
                except Exception:
                    fig = None

        return llm_json, fig, node_df, (link_df if isinstance(link_df, pd.DataFrame) else pd.DataFrame())

    return {"result": {"error": "NOT_IMPLEMENTED"}}, None, pd.DataFrame(), pd.DataFrame()


def _answer_graph_stats(query: str) -> Dict[str, Any]:
    import networkx as nx

    ql = (query or "").lower()
    deg = dict(net.G.degree())
    if not deg:
        return {"result": {"error": "EMPTY_TOPOLOGY"}}

    if "highest degree" in ql:
        n = max(deg, key=lambda k: deg[k])
        return {"result": {"node_id": str(n), "degree": int(deg[n])}}

    if "lowest degree" in ql:
        n = min(deg, key=lambda k: deg[k])
        return {"result": {"node_id": str(n), "degree": int(deg[n])}}

    if "average node degree" in ql:
        vals = list(deg.values())
        return {
            "result": {
                "average_degree": float(sum(vals) / len(vals)),
                "min_degree": int(min(vals)),
                "max_degree": int(max(vals)),
            }
        }

    if "degree 1" in ql:
        return {"result": {"value": int(sum(1 for v in deg.values() if v == 1))}}

    if "degree >=" in ql:
        m = re.search(r"degree\s*>=\s*(\d+)", ql)
        k = int(m.group(1)) if m else 3
        return {"result": {"value": int(sum(1 for v in deg.values() if v >= k))}}

    if "isolated" in ql:
        return {"result": {"value": bool(any(v == 0 for v in deg.values()))}}

    if "top 3" in ql:
        top = sorted(deg.items(), key=lambda kv: (-kv[1], str(kv[0])))[:3]
        return {"result": {"top_nodes": [{"node_id": str(n), "degree": int(d)} for n, d in top]}}

    if "diameter" in ql:
        if nx.is_connected(net.G):
            return {"result": {"value": int(nx.diameter(net.G))}}
        comps = [net.G.subgraph(c).copy() for c in nx.connected_components(net.G)]
        d = max(int(nx.diameter(g)) for g in comps if g.number_of_nodes() > 1)
        return {"result": {"value": d}}

    if "average shortest-path length" in ql:
        if nx.is_connected(net.G):
            return {"result": {"value": float(nx.average_shortest_path_length(net.G))}}
        largest = max(nx.connected_components(net.G), key=len)
        g = net.G.subgraph(largest).copy()
        return {"result": {"value": float(nx.average_shortest_path_length(g))}}

    return {"result": {"error": "NOT_IMPLEMENTED"}}


def _answer_link_stats(query: str, flags: Dict[str, bool]) -> Dict[str, Any]:
    ql = (query or "").lower()
    df = _edges_df()

    m = re.search(r"\b(ml|al)\s+links\s+under\s+(\d+(?:\.\d+)?)\s*km\b", ql)
    if m:
        typ = m.group(1).upper()
        th = float(m.group(2))
        f = df[
            (df["type"].astype(str).str.upper() == typ)
            & (pd.to_numeric(df["distance_km"], errors="coerce") < th)
        ]
        rows = (
            f[["u", "v", "link_id", "type", "distance_km"]]
            .sort_values("distance_km")
            .to_dict("records")
        )
        return {
            "result": {"value": int(len(rows))},
            "tables": [{"name": f"{typ}_links_under_{th}_km", "rows": rows}],
        }

    m = re.search(r"\baverage\s+(ml|al)\s+link\s+distance\b", ql)
    if m:
        typ = m.group(1).upper()
        f = df[df["type"].astype(str).str.upper() == typ]
        vals = pd.to_numeric(f["distance_km"], errors="coerce").dropna()
        return {"result": {"value": float(vals.mean()) if len(vals) else None}}

    if "maximum ml link distance" in ql or ("max" in ql and "ml" in ql and "distance" in ql):
        f = df[df["type"].astype(str).str.upper() == "ML"].copy()
        f["distance_km"] = pd.to_numeric(f["distance_km"], errors="coerce")
        f = f.dropna(subset=["distance_km"])
        if f.empty:
            return {"result": {"error": "NO_ML_EDGES"}}
        row = f.sort_values("distance_km", ascending=False).iloc[0].to_dict()
        out: Dict[str, Any] = {"result": row}
        if flags["viz_any"]:
            out["plot"] = {
                "title": "Max ML link",
                "highlight_edges": [[row["u"], row["v"]]],
                "highlight_nodes": [row["u"], row["v"]],
            }
        return out

    if "5 longest links" in ql:
        f = df.copy()
        f["distance_km"] = pd.to_numeric(f["distance_km"], errors="coerce")
        f = f.dropna(subset=["distance_km"])
        top = f.sort_values("distance_km", ascending=False).head(5)[
            ["u", "v", "link_id", "type", "distance_km"]
        ]
        rows = top.to_dict("records")
        return {
            "result": {"value": int(len(rows))},
            "tables": [{"name": "top_5_longest_links", "rows": rows}],
        }

    return {"result": {"error": "NOT_IMPLEMENTED"}}


def _answer_equipment(query: str) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ql = (query or "").lower()

    eqp_mngr.deploy_equipment()
    s = eqp_mngr.summarize_equipment(visualize=False)
    node_df = s.get("node_df")
    link_df = s.get("link_df")

    if not isinstance(node_df, pd.DataFrame):
        node_df = pd.DataFrame()
    if not isinstance(link_df, pd.DataFrame):
        link_df = pd.DataFrame()

    def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = [c for c in df.columns]
        for cand in candidates:
            for c in cols:
                if cand in c.lower():
                    return c
        return None

    node_eq_col = _find_col(node_df, ["equipment", "elements", "deployed"])
    link_eq_col = _find_col(link_df, ["equipment", "elements", "deployed"])

    # Always return proper tables for "summary table" requests
    if "summary table" in ql or ("deploy network equipment" in ql and "table" in ql):
        return (
            {
                "result": {"value": "EQUIPMENT_DEPLOYED"},
                "tables": [
                    {"name": "node_equipment", "rows": node_df.to_dict("records")},
                    {"name": "link_equipment", "rows": link_df.to_dict("records")},
                ],
            },
            node_df,
            link_df,
        )

    if "how many roadm" in ql:
        if node_eq_col is None:
            return ({"result": {"error": "NO_NODE_EQUIPMENT_COLUMN"}}, node_df, link_df)
        cnt = int(
            node_df[node_eq_col]
            .astype(str)
            .str.contains("ROADM", case=False, na=False)
            .sum()
        )
        return ({"result": {"value": cnt}}, node_df, link_df)

    if "how many trx" in ql or "trx elements" in ql:
        if node_eq_col is None:
            return ({"result": {"error": "NO_NODE_EQUIPMENT_COLUMN"}}, node_df, link_df)
        cnt = int(
            node_df[node_eq_col]
            .astype(str)
            .str.contains("TRX|TRANSCEIVER", case=False, na=False, regex=True)
            .sum()
        )
        return ({"result": {"value": cnt}}, node_df, link_df)

    if "total number of ilas" in ql or "total number of ila" in ql:
        if link_eq_col is None:
            return ({"result": {"error": "NO_LINK_EQUIPMENT_COLUMN"}}, node_df, link_df)
        cnt = int(link_df[link_eq_col].astype(str).str.count("ILA").sum())
        return ({"result": {"value": cnt}}, node_df, link_df)

    def _links_with(token: str, table_name: str) -> Dict[str, Any]:
        if link_eq_col is None:
            return {"result": {"error": "NO_LINK_EQUIPMENT_COLUMN"}}
        f = link_df[
            link_df[link_eq_col].astype(str).str.contains(token, case=False, na=False)
        ].copy()
        rows = f.to_dict("records")
        return {"result": {"value": int(len(rows))}, "tables": [{"name": table_name, "rows": rows}]}

    if "links that have an ila" in ql:
        return (_links_with("ILA", "links_with_ila"), node_df, link_df)

    if "links that have a preamp" in ql:
        return (_links_with("PREAMP", "links_with_preamp"), node_df, link_df)

    if "links that have an amp" in ql:
        if link_eq_col is None:
            return ({"result": {"error": "NO_LINK_EQUIPMENT_COLUMN"}}, node_df, link_df)
        f = link_df[
            link_df[link_eq_col].astype(str).str.contains("AMP", case=False, na=False)
            & ~link_df[link_eq_col].astype(str).str.contains("PREAMP", case=False, na=False)
        ].copy()
        rows = f.to_dict("records")
        return (
            {"result": {"value": int(len(rows))}, "tables": [{"name": "links_with_amp", "rows": rows}]},
            node_df,
            link_df,
        )

    # Default: still return tables so downstream app/bench can display them
    return (
        {
            "result": {"value": "EQUIPMENT_DEPLOYED"},
            "tables": [
                {"name": "node_equipment", "rows": node_df.to_dict("records")},
                {"name": "link_equipment", "rows": link_df.to_dict("records")},
            ],
        },
        node_df,
        link_df,
    )


# ------------------------------ LLM fallback prompt ------------------------------
_SYSTEM_PROMPT = """
    You are an assistant for optical network analytics.

OUTPUT FORMAT (HARD RULE):
- Return exactly ONE valid JSON object.
- No markdown, no code fences, no extra text.
- Allowed top-level keys ONLY: result, tables, plot.
- 'plot' MUST be top-level (never nested inside result).

FAIL-INSTEAD-OF-GUESS (HARD RULE):
- If you cannot answer exactly from the provided payload, return:
  { "result": { "error": "NOT_IMPLEMENTED" } }

INTERPRETATION PRIORITY (HARD RULE):
- If the user asks for "shortest path" or "longest path", you MUST produce a path result (or NOT_IMPLEMENTED).
- If the user also asks to plot, you MUST provide a plot that matches the requested scope.

PLOT CONTRACT:
- The plot object must be:
  { "plot": { "title": "...", "highlight_edges": "ALL" or [[u,v],...], "highlight_nodes": "ALL" or [..] } }
- Do NOT include raw topology data inside plot.

PATH-ONLY VISUALIZATION RULE (HARD RULE):
- If the user says ANY of: "plot that path only", "only the path", "not the full topology", "do not show full topology",
  then you MUST NOT use "ALL" for highlight_edges or highlight_nodes.
- Instead, highlight_edges MUST be an explicit edge list [[u,v], ...] for the chosen path.
- highlight_nodes MUST be the ordered node list of the chosen path.

TOPOLOGY VISUALIZATION RULE:
- Use "ALL" ONLY when the user explicitly asks for full topology / entire network / full graph / all links.

DEFAULT TITLE RULES:
- If shortest path: title = "Shortest Path: <SRC> → <DST>"
- If longest path:  title = "Longest Path: <SRC> → <DST>"
- If full topology: title = "Network topology"
"""


def build_messages(query: str) -> List[Dict[str, str]]:
    payload = {
        "user_query": query,
        "topology": net.topology_snapshot(),
        "note": "Prefer returning NOT_IMPLEMENTED rather than guessing.",
    }
    return [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(payload)}]


# ------------------------------------ run() ------------------------------------
def run(query: str, mode: str = "none", rag_enabled: bool = False) -> Dict[str, Any]:
    global _TOPO_READY

    if not _TOPO_READY:
        net.build_topology(use_csv=True, visualize=False)
        _TOPO_READY = True

    flags = _query_flags(query)

    llm_json: Dict[str, Any] = {"result": {"error": "NOT_IMPLEMENTED"}}
    fig = None
    node_df = link_df = qot_df = None
    prompt_tokens = completion_tokens = total_tokens = None
    raw = ""
    ok = True

    try:
        if flags["equip"]:
            llm_json, node_df, link_df = _answer_equipment(query)
            raw = json.dumps(llm_json)

        elif flags["path"]:
            llm_json, fig, node_df, link_df = _answer_paths(query, flags)
            raw = json.dumps(llm_json)


        elif flags["topology"]:
            llm_json, fig = _answer_topology(flags)
            raw = json.dumps(llm_json)


        elif flags["graph_stats"]:
            llm_json = _answer_graph_stats(query)
            raw = json.dumps(llm_json)

        elif flags["link_stats"]:
            llm_json = _answer_link_stats(query, flags)
            raw = json.dumps(llm_json)

        else:
            messages = build_messages(query)
            raw, prompt_tokens, completion_tokens, total_tokens = call_llm(messages)
            llm_json = _extract_first_json(raw)

    except Exception as e:
        ok = False
        llm_json = {"result": {"error": "PARSE_ERROR", "message": str(e), "raw": (raw or "")[:2000]}}
        raw = json.dumps(llm_json)

    # If we didn't already compute fig directly, render via plot spec
    if fig is None:
        plot = llm_json.get("plot") if isinstance(llm_json, dict) else None
        if isinstance(plot, dict):
            try:
                _, fig = net.render_plot(plot)
            except Exception:
                fig = None

    # Tables into DFs (so app + benchmark can show them)
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
        "ok": ok and isinstance(llm_json, dict),
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
