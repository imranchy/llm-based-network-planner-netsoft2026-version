"""llm_agent.py

Hybrid deterministic network analytics + strict-JSON LLM fallback.

Deterministic handlers cover:
- topology build from CSV (+ optional visualization)
- shortest path (Dijkstra), fewest hops (BFS)
- longest path policy: longest simple path within hop cutoff
  * If user doesn't specify cutoff, uses max(12, 2 * shortest_hops)
- bounded all-paths listing (within hop cutoff)
- graph stats (degree stats, isolated, average shortest-path length in hops)
- link stats (ML/AL filters, averages, max, top-k)
- equipment deployment + summaries and counts
- propagation delay & fiber-type comparisons (uses qot_utils.calc_qot_metrics)
- ILA-constrained routing (min ILAs, paths containing edges with ILA threshold)
- HCF upgrade recommendation (heuristic)

The LLM is used only as a strict-JSON fallback for unsupported queries.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import networkx as nx
try:
    import ollama  # type: ignore
except Exception:  # ollama python package may be missing
    ollama = None
import pandas as pd
import psutil

from equipment_manager import EquipmentManager
from graph_processor import NetworkGraph
from qot_utils import calc_qot_metrics
from graph_utils import resolve_nodes

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
net = NetworkGraph(os.path.join(BASE_DIR,"distance.csv"), os.path.join(BASE_DIR,"city.csv"))
eqp_mngr = EquipmentManager(net)
_TOPO_READY = True
_EQUIP_READY = False

def _reset_state() -> None:
    global _TOPO_READY, _EQUIP_READY, net, eqp_mngr
    net = NetworkGraph(os.path.join(BASE_DIR,"distance.csv"), os.path.join(BASE_DIR,"city.csv"))
    eqp_mngr = EquipmentManager(net)
    _TOPO_READY = True
    _EQUIP_READY = False

def _ensure_equipment() -> None:
    global _EQUIP_READY
    if not _EQUIP_READY:
        eqp_mngr.deploy_equipment()
        _EQUIP_READY = True

# --------------------------- CPU / RAM / GPU logging (optional) ---------------------------
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
_DEF_SEED = 42

def _decode_options() -> dict:
    return {"seed": _DEF_SEED, "top_p": 1.0, "temperature": 0.0, "num_predict": 700}

def call_llm(messages: List[Dict[str, str]]) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    """Call Ollama.

    Uses the `ollama` Python package when available; otherwise falls back to the local HTTP API
    (default Ollama endpoint: http://127.0.0.1:11434).
    """
    opts = _decode_options()

    # Preferred: python package
    if ollama is not None:
        r = ollama.chat(model=_ACTIVE_MODEL, messages=messages, options=opts)
        content = (r["message"]["content"] or "").strip()
        prompt_tokens = r.get("prompt_eval_count")
        completion_tokens = r.get("eval_count")
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        return content, prompt_tokens, completion_tokens, total_tokens

    # Fallback: local HTTP API
    import json as _json
    import urllib.request

    payload = _json.dumps(
        {"model": _ACTIVE_MODEL, "messages": messages, "options": opts, "stream": False}
    ).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = _json.loads(resp.read().decode("utf-8"))

    msg = (data.get("message") or {}).get("content") or ""
    content = str(msg).strip()

    # HTTP API may expose token counts under `prompt_eval_count` / `eval_count` as well
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

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
# ------------------------------ LLM planning (LLM-first) ------------------------------
# The model decides the intent + parameters, then Python executes deterministically.

_INTENTS = {
    "TOPOLOGY_BUILD",
    "SHORTEST_PATH",
    "FEWEST_HOPS",
    "LONGEST_PATH",
    "ALL_PATHS",
    "GRAPH_STATS",
    "LINK_STATS",
    "EQUIPMENT",
    "PROP_DELAY",
    "FIBER_CHOICE",
    "HCF_UPGRADE",
    "ILA_MIN_ROUTE",
    "ILA_THRESH_PATHS",
}

def _llm_disabled() -> bool:
    return str(os.environ.get("OLLAMA_DISABLE", "")).strip().lower() in {"1","true","yes","on"}

def llm_plan(query: str) -> Optional[Dict[str, Any]]:
    """Ask the selected Ollama model for a strict JSON plan."""
    if _llm_disabled():
        return None

    sys = (
        "You are a router for a network analytics app. "
        "Return ONLY one JSON object (no markdown, no backticks). "
        "Choose intent from this list: "
        + ", ".join(sorted(_INTENTS))
        + ". "
        "Extract src/dst nodes for path questions. "
        "Set visualize=true if user asks to plot/draw/visualize; otherwise false. "
        "Set path_only=true only when user asks to plot only the path. "
        "If a question is about ML/AL links, set link_type to 'ML' or 'AL'. "
        "If a threshold is mentioned (e.g., ILAs >= 4, links under 5 km), set threshold/distance_km. "
        "If top-k is requested, set k."
    )
    user = "Question: " + (query or "").strip()

    try:
        raw, *_ = call_llm([{"role":"system","content":sys},{"role":"user","content":user}])
        plan = _extract_first_json(raw)
        if not isinstance(plan, dict):
            return None
        intent = str(plan.get("intent","")).strip().upper()
        if intent not in _INTENTS:
            return None
        for bkey in ("visualize","path_only","want_table"):
            if bkey in plan:
                plan[bkey] = bool(plan[bkey])
        if "link_type" in plan and plan["link_type"] is not None:
            lt = str(plan["link_type"]).upper().strip()
            if lt in {"ML","AL"}:
                plan["link_type"] = lt
        return plan
    except Exception:
        return None

def _query_flags(q: str) -> Dict[str, bool]:
    ql = (q or "").lower()

    no_plot = bool(re.search(r"\b(no plot|no visualization|without visualization)\b", ql))
    viz_any = bool(re.search(r"\b(plot|draw|visuali[sz]e|render)\b", ql)) and not no_plot

    # Normalize unicode comparisons
    ql_norm = ql.replace("≥", ">=").replace("≤", "<=")

    return {
        "no_plot": no_plot,
        "viz_any": viz_any,
        "path_only": bool(re.search(r"\b(path only|only the path|plot that path only|not the full topology|not the topology)\b", ql_norm)),

        "topology": bool(re.search(r"\b(create|build|draw|visualize)\b.*\b(topology|network)\b|\bfrom csv\b|\bcsv files?\b", ql_norm)),

        "path": bool(re.search(r"\b(shortest path|longest path|fewest hops|list all paths|all paths)\b", ql_norm)),

        "graph_stats": bool(re.search(r"(highest degree|lowest degree|average node degree|degree 1|degree\s*(?:>=|>|=|≥)\s*\d+|isolated|top 3|average shortest[- ]path length|diameter)", ql_norm)),

        "link_stats": bool(re.search(r"\b(ml links|al links|average .* link distance|average ml|average al|max(?:imum)? ml|longest links|\d+ longest links|top \d+ longest)\b", ql_norm)),

        "equip": bool(re.search(r"\b(deploy|equipment|roadm|trx|ila|preamp|amp)\b", ql_norm)),

        "delay": bool(re.search(r"\b(propagation delay|delay|latency)\b", ql_norm)),
        "fiber_choice": bool(re.search(r"\b(which fiber|fiber type|min(imum)? delay|hcf|ssmf)\b", ql_norm)),
        "hcf_upgrade": bool(re.search(r"\bhcf\b.*\b(only|\d+)\b.*\blinks\b|\bdeploy\s+hcf\b|\bplace\s+hcf\b", ql_norm)),

        "ila_min": bool(re.search(r"\b(minimum(?:\s+number\s+of)?\s+ilas|minimize\s+ilas|fewest\s+ilas|min\s+ilas)\b", ql_norm)),
        "ila_thresh": bool(re.search(r"\b(?:ilas|ila)\s*(?:>=|≥)\s*\d+\b", ql_norm)),

        "distance_labels": bool(re.search(r"\b(label each link|label each edge|with distances)\b", ql_norm)),
    }
def _clean_endpoint(s: str) -> str:
    s = (s or "").strip().strip('"').strip("'").strip()
    s = s.split(",")[0].strip()
    s = re.split(r"\s*[\(\[]", s, maxsplit=1)[0].strip()
    s = re.split(r"\s+and\s+(?:plot|draw|render|visuali[sz]e|show|return|list|count|compute|calculate)\b", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    s = re.split(r"\s+(?:plot|draw|render|visuali[sz]e|show|return|list|count|compute|calculate)\b", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    s = re.split(r"\s+by\s+.*$", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return s.strip(" .?;,:")

def extract_src_dst(query: str) -> Tuple[Optional[str], Optional[str]]:
    q = (query or "").strip()

    # from X to Y ...
    m = re.search(
        r"\bfrom\s+(.+?)\s+to\s+(.+?)(?=\s+and\s+(?:plot|draw|render|visuali[sz]e|show|return|list|count|compute|calculate)\b|[\?\.]?$|\s*$)",
        q,
        flags=re.IGNORECASE,
    )
    if m:
        a = _clean_endpoint(m.group(1))
        b = _clean_endpoint(m.group(2))
        if re.fullmatch(r"M\d+", a, flags=re.IGNORECASE):
            a = a.upper()
        if re.fullmatch(r"M\d+", b, flags=re.IGNORECASE):
            b = b.upper()
        return a, b

    # between X and Y ... [stop before "and plot/return/..."]
    m = re.search(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?=\s+and\s+(?:plot|draw|render|visuali[sz]e|show|return|list|count|compute|calculate)\b|[\?\.]?$|\s*$)",
        q,
        flags=re.IGNORECASE,
    )
    if m:
        a = _clean_endpoint(m.group(1))
        b = _clean_endpoint(m.group(2))
        if re.fullmatch(r"M\d+", a, flags=re.IGNORECASE):
            a = a.upper()
        if re.fullmatch(r"M\d+", b, flags=re.IGNORECASE):
            b = b.upper()
        return a, b

    # fallback: match any node names mentioned in query
    ql = q.lower()
    hits: List[str] = []
    for n in net.G.nodes():
        ns = str(n)
        if ns.lower() in ql:
            hits.append(ns)
    if len(hits) >= 2:
        return hits[0], hits[1]

    return None, None

def _parse_max_hops(query: str) -> Optional[int]:
    ql = (query or "").lower().replace("≥", ">=")
    m = re.search(r"\b(?:within|max(?:imum)?|at most)\s*(\d{1,3})\s*hops\b", ql)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _path_edges(path: List[Any]) -> List[List[str]]:
    if not isinstance(path, list) or len(path) < 2:
        return []
    return [[str(path[i]), str(path[i + 1])] for i in range(len(path) - 1)]

def _path_total_distance_km(path: List[str]) -> float:
    total = 0.0
    for u, v in zip(path, path[1:]):
        data = net.G.get_edge_data(u, v) or {}
        w = data.get("weight", 0.0)
        try:
            total += float(w)
        except Exception:
            pass
    return float(total)

def _edges_df() -> pd.DataFrame:
    rows = []
    for u, v, d in net.G.edges(data=True):
        link_id = str(d.get("link_id") or "")
        dist = d.get("weight")
        try:
            dist = float(dist)
        except Exception:
            dist = None
        typ = str(d.get("type") or "").upper()
        inferred = ""
        m = re.match(r"^(ML|AL)", link_id.upper())
        if m:
            inferred = m.group(1)
        if typ not in {"ML", "AL"}:
            typ = inferred or typ
        rows.append({"u": str(u), "v": str(v), "link_id": link_id, "type": typ, "distance_km": dist})
    return pd.DataFrame(rows)

# ------------------------------ Deterministic handlers ------------------------------
def _answer_topology(flags: Dict[str, bool]) -> Tuple[Dict[str, Any], Any]:
    out: Dict[str, Any] = {"result": {"value": "TOPOLOGY_READY"}}
    fig = None
    if flags["viz_any"] and not flags["no_plot"]:
        out["plot"] = {
            "title": "Network topology",
            "highlight_edges": "ALL",
            "highlight_nodes": "ALL",
            "only_highlight": False,
            "show_edge_labels": bool(flags.get("distance_labels", False)),
            "label_font_size": 6,
            "edge_label_font_size": 6,
        }
        _, fig = net.render_plot(out["plot"])
    return out, fig

def _answer_paths(query: str, flags: Dict[str, bool]) -> Tuple[Dict[str, Any], Any, pd.DataFrame, pd.DataFrame]:
    ql = (query or "").lower().replace("≥", ">=")
    src, dst = extract_src_dst(query)
    if not src or not dst:
        return {"result": {"error": "NO_PATH", "message": "Could not extract SRC/DST from query."}}, None, pd.DataFrame(), pd.DataFrame()

    a, b = resolve_nodes(net.G, src, dst)

    mode: Optional[str] = None
    if re.search(r"\bshortest\b", ql):
        mode = "shortest"
    elif re.search(r"\bfewest\s+hops\b", ql):
        mode = "fewest_hops"
    elif re.search(r"\blongest\b", ql):
        mode = "longest"
    elif re.search(r"\blist\s+all\s+paths\b|\ball\s+paths\b", ql):
        mode = "all"

    if mode is None:
        return {"result": {"error": "NOT_IMPLEMENTED"}}, None, pd.DataFrame(), pd.DataFrame()

    fig = None

    if mode == "shortest":
        try:
            path = nx.dijkstra_path(net.G, source=a, target=b, weight="weight")
        except Exception:
            return {"result": {"error": "NO_PATH", "message": f"No path found between {src} and {dst}."}}, None, pd.DataFrame(), pd.DataFrame()

        total_km = _path_total_distance_km(path)
        out = {"result": {"path": path, "hop_count": len(path) - 1, "total_distance_km": round(total_km, 6)}}

        if flags["viz_any"] and not flags["no_plot"]:
            out["plot"] = {
                "title": f"Shortest Path: {src} → {dst}",
                "highlight_nodes": path,
                "highlight_edges": _path_edges(path),
                "only_highlight": True,
                "path_only": True,
                "highlight_only": True,
                "show_edge_labels": False,
                "label_font_size": 7,
            }
            _, fig = net.render_plot(out["plot"])
        return out, fig, pd.DataFrame([{"Nodes": path, "Hop Count": len(path) - 1, "Total Distance (km)": total_km}]), pd.DataFrame()

    if mode == "fewest_hops":
        try:
            path = nx.shortest_path(net.G, source=a, target=b)
        except Exception:
            return {"result": {"error": "NO_PATH", "message": f"No path found between {src} and {dst}."}}, None, pd.DataFrame(), pd.DataFrame()
        total_km = _path_total_distance_km(path)
        out = {"result": {"path": path, "hop_count": len(path) - 1}}
        if flags["viz_any"] and not flags["no_plot"]:
            out["plot"] = {
                "title": f"Fewest Hops: {src} → {dst}",
                "highlight_nodes": path,
                "highlight_edges": _path_edges(path),
                "only_highlight": True,
                "path_only": True,
                "highlight_only": True,
                "show_edge_labels": False,
                "label_font_size": 7,
            }
            _, fig = net.render_plot(out["plot"])
        return out, fig, pd.DataFrame([{"Nodes": path, "Hop Count": len(path) - 1, "Total Distance (km)": total_km}]), pd.DataFrame()

    # longest / all: bounded simple paths
    user_hops = _parse_max_hops(query)
    try:
        shortest_hops = len(nx.shortest_path(net.G, source=a, target=b)) - 1
    except Exception:
        return {"result": {"error": "NO_PATH", "message": f"No path found between {src} and {dst}."}}, None, pd.DataFrame(), pd.DataFrame()

    max_hops = user_hops if user_hops is not None else max(12, 2 * shortest_hops)

    # enumerate bounded simple paths
    paths = []
    for p in nx.all_simple_paths(net.G, source=a, target=b, cutoff=max_hops):
        paths.append(list(p))

    if not paths:
        return {"result": {"error": "NO_PATH", "message": f"No paths found within {max_hops} hops."}}, None, pd.DataFrame(), pd.DataFrame()

    if mode == "all":
        out = {"result": {"count": len(paths), "max_hops": max_hops, "paths": paths}}
        return out, None, pd.DataFrame(), pd.DataFrame()

    # mode == longest
    best = None
    best_dist = None
    for p in paths:
        d = _path_total_distance_km(p)
        if best_dist is None or d > best_dist:
            best_dist = d
            best = p

    assert best is not None and best_dist is not None
    out = {
        "result": {
            "path": best,
            "hop_count": len(best) - 1,
            "total_distance_km": round(float(best_dist), 6),
            "max_hops": int(max_hops),
            "note": "Longest path computed among simple paths with hop_count <= max_hops.",
        }
    }
    if flags["viz_any"] and not flags["no_plot"]:
        out["plot"] = {
            "title": f"Longest (≤{max_hops} hops): {src} → {dst}",
            "highlight_nodes": best,
            "highlight_edges": _path_edges(best),
            "only_highlight": True,
            "path_only": True,
            "highlight_only": True,
            "show_edge_labels": False,
            "label_font_size": 7,
        }
        _, fig = net.render_plot(out["plot"])
    return out, fig, pd.DataFrame([{"Nodes": best, "Hop Count": len(best) - 1, "Total Distance (km)": best_dist}]), pd.DataFrame()

def _answer_graph_stats(query: str) -> Dict[str, Any]:
    ql = (query or "").lower().replace("≥", ">=")
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
        return {"result": {"average_degree": float(sum(vals) / len(vals)), "min_degree": int(min(vals)), "max_degree": int(max(vals))}}

    if "degree 1" in ql:
        return {"result": {"value": int(sum(1 for v in deg.values() if v == 1))}}

    m = re.search(r"degree\s*(?:>=|≥)\s*(\d+)", ql)
    if m:
        k = int(m.group(1))
        return {"result": {"value": int(sum(1 for v in deg.values() if v >= k))}}

    if "isolated" in ql:
        return {"result": {"value": bool(any(v == 0 for v in deg.values()))}}

    if "top 3" in ql:
        top = sorted(deg.items(), key=lambda kv: (-kv[1], str(kv[0])))[:3]
        return {"result": {"top3": [{"node_id": str(n), "degree": int(d)} for n, d in top]}}

    if "average shortest" in ql:
        # hops; use largest CC if disconnected
        if nx.is_connected(net.G):
            asp = nx.average_shortest_path_length(net.G)
            return {"result": {"average_shortest_path_length_hops": float(asp)}}
        comps = sorted(nx.connected_components(net.G), key=len, reverse=True)
        g = net.G.subgraph(comps[0]).copy()
        asp = nx.average_shortest_path_length(g)
        return {"result": {"average_shortest_path_length_hops": float(asp), "note": "Graph disconnected; computed on largest connected component."}}

    return {"result": {"error": "NOT_IMPLEMENTED"}}

def _answer_link_stats(query: str, flags: Dict[str, bool]) -> Tuple[Dict[str, Any], Any, pd.DataFrame]:
    ql = (query or "").lower().replace("≥", ">=")
    df = _edges_df()
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    fig = None

    m = re.search(r"\b(ml|al)\s+links\s+under\s+(\d+(?:\.\d+)?)\s*km\b", ql)
    if m:
        lt = m.group(1).upper()
        th = float(m.group(2))
        sub = df[df["type"].str.upper().eq(lt) & (df["distance_km"] <= th)].copy()
        out = {"result": {"count": int(len(sub))}, "tables": {"links": sub[["u","v","link_id","distance_km","type"]].to_dict(orient="records")}}
        return out, None, sub

    m = re.search(r"\baverage\s+(ml|al)\b.*\bdistance\b", ql)
    if m:
        lt = m.group(1).upper()
        sub = df[df["type"].str.upper().eq(lt)]
        if sub.empty:
            return {"result": {"value": 0, "note": f"No {lt} links found in the CSV topology."}}, None, pd.DataFrame()
        avg_val = sub["distance_km"].mean()
        if pd.isna(avg_val):
            return {"result": {"value": 0, "note": f"No {lt} links found in the CSV topology."}}, None, pd.DataFrame()
        return {"result": {"value": float(avg_val)}}, None, sub

    if re.search(r"\b(max(?:imum)?\s+ml|max\s+ml)\b", ql):
        sub = df[df["type"].str.upper().eq("ML")]
        if sub.empty:
            return {"result": {"value": 0, "note": "No ML links found in the CSV topology."}}, None, pd.DataFrame()
        idx = sub["distance_km"].idxmax()
        row = sub.loc[idx]
        u, v = str(row["u"]), str(row["v"])
        out = {"result": {"u": u, "v": v, "link_id": row.get("link_id"), "distance_km": float(row["distance_km"])}}
        if flags["viz_any"] and not flags["no_plot"]:
            out["plot"] = {"title": "Max ML link", "highlight_edges": [[u, v]], "highlight_nodes": [u, v], "only_highlight": True, "highlight_only": True, "show_edge_labels": True, "label_font_size": 7, "edge_label_font_size": 7}
            _, fig = net.render_plot(out["plot"])
        return out, fig, sub

    m = re.search(r"\b(\d+)\s+longest\s+links\b", ql)
    if m:
        k = int(m.group(1))
        sub = df.dropna(subset=["distance_km"]).sort_values("distance_km", ascending=False).head(k)
        return {"result": {"k": k, "links": sub[["u","v","link_id","distance_km","type"]].to_dict(orient="records")}}, None, sub

    if "5 longest links" in ql or ("longest links" in ql and "5" in ql):
        sub = df.dropna(subset=["distance_km"]).sort_values("distance_km", ascending=False).head(5)
        return {"result": {"k": 5, "links": sub[["u","v","link_id","distance_km","type"]].to_dict(orient="records")}}, None, sub

    return {"result": {"error": "NOT_IMPLEMENTED"}}, None, pd.DataFrame()

def _answer_equipment(query: str) -> Tuple[Dict[str, Any], Any, pd.DataFrame, pd.DataFrame]:
    _ensure_equipment()
    ql = (query or "").lower().replace("≥", ">=")

    # summary table (only when explicitly asked)
    if "summary" in ql or "show a summary" in ql:
        summary = eqp_mngr.summarize_equipment(visualize=False)
        return {"result": {"value": "EQUIPMENT_READY"}}, None, summary.get("node_df", pd.DataFrame()), summary.get("link_df", pd.DataFrame())

    # counts
    if "how many roadm" in ql or "roadms" in ql:
        n = sum(1 for _, d in net.G.nodes(data=True) for e in (d.get("equipment_list") or []) if str(e).startswith("ROADM_"))
        return {"result": {"value": int(n)}}, None, pd.DataFrame(), pd.DataFrame()

    if "how many trx" in ql or "trx" in ql:
        n = sum(1 for _, d in net.G.nodes(data=True) for e in (d.get("equipment_list") or []) if str(e).startswith("TRx_"))
        return {"result": {"value": int(n)}}, None, pd.DataFrame(), pd.DataFrame()

    if "total number of ilas" in ql or "count the total number of ilas" in ql:
        total = int(sum(int(d.get("ila_count", 0) or 0) for _, _, d in net.G.edges(data=True)))
        return {"result": {"value": total}}, None, pd.DataFrame(), pd.DataFrame()

    def _links_with(prefix: str) -> List[Dict[str, Any]]:
        rows = []
        for u, v, d in net.G.edges(data=True):
            eqs = d.get("equipment_list") or []
            if isinstance(eqs, str):
                eqs = [x.strip() for x in eqs.split("|") if x.strip()]
            if any(str(e).startswith(prefix) for e in eqs):
                rows.append({"u": str(u), "v": str(v), "equipment": " | ".join(map(str, eqs))})
        return rows

    if "links that have an ila" in ql:
        rows = _links_with("ILA_")
        return {"result": {"count": len(rows)}, "tables": {"links": rows}}, None, pd.DataFrame(), pd.DataFrame()

    if "links that have a preamp" in ql:
        rows = _links_with("PreAmp_")
        return {"result": {"count": len(rows)}, "tables": {"links": rows}}, None, pd.DataFrame(), pd.DataFrame()

    if "links that have an amp" in ql and "preamp" not in ql:
        rows = _links_with("Amp_")
        return {"result": {"count": len(rows)}, "tables": {"links": rows}}, None, pd.DataFrame(), pd.DataFrame()

    return {"result": {"value": "EQUIPMENT_READY"}}, None, pd.DataFrame(), pd.DataFrame()

def _answer_propagation_delay(query: str, flags: Dict[str, bool]) -> Tuple[Dict[str, Any], Any]:
    src, dst = extract_src_dst(query)
    if not src or not dst:
        return {"result": {"error": "NO_PATH", "message": "Could not extract SRC/DST from query."}}, None
    a, b = resolve_nodes(net.G, src, dst)
    try:
        path = nx.dijkstra_path(net.G, source=a, target=b, weight="weight")
    except Exception:
        return {"result": {"error": "NO_PATH"}}, None
    total_km = _path_total_distance_km(path)
    metrics = calc_qot_metrics(total_km, fiber_type="SSMF", optical_params=None)
    delay_ms = float(metrics.get("Total Propagation Delay (ms)", 0.0) or 0.0)
    # Fallback: if qot_utils returns 0/null, approximate using v≈2e8 m/s (~5 µs/km => 0.005 ms/km)
    if (delay_ms is None or delay_ms == 0.0) and total_km and total_km > 0:
        delay_ms = float(total_km) * 0.005
    out = {"result": {"path": path, "hop_count": len(path)-1, "total_distance_km": round(total_km,6), "propagation_delay_ms": round(delay_ms,6)}}
    fig = None
    if flags["viz_any"] and not flags["no_plot"]:
        out["plot"] = {"title": f"Propagation delay path: {src} → {dst}", "highlight_nodes": path, "highlight_edges": _path_edges(path), "only_highlight": True, "path_only": True, "highlight_only": True, "show_edge_labels": False, "label_font_size": 7}
        _, fig = net.render_plot(out["plot"])
    return out, fig

def _answer_fiber_min_delay(query: str) -> Dict[str, Any]:
    src, dst = extract_src_dst(query)
    if not src or not dst:
        return {"result": {"error": "NO_PATH", "message": "Could not extract SRC/DST from query."}}
    a, b = resolve_nodes(net.G, src, dst)
    try:
        path = nx.dijkstra_path(net.G, source=a, target=b, weight="weight")
    except Exception:
        return {"result": {"error": "NO_PATH"}}
    total_km = _path_total_distance_km(path)
    candidates = ["SSMF", "HCF"]
    rows = []
    best_ft = None
    best_delay = None
    for ft in candidates:
        m = calc_qot_metrics(total_km, fiber_type=ft, optical_params=None)
        d = float(m.get("Total Propagation Delay (ms)", 0.0) or 0.0)
        if (d is None or d == 0.0) and total_km and total_km > 0:
            # same baseline approximation; HCF assumed slightly faster (2% here) if qot_utils missing
            baseline = float(total_km) * 0.005
            d = baseline * (0.98 if ft == "HCF" else 1.0)
        rows.append({"fiber_type": ft, "propagation_delay_ms": round(d,6)})
        if best_delay is None or d < best_delay:
            best_delay = d
            best_ft = ft
    return {"result": {"src": src, "dst": dst, "total_distance_km": round(total_km,6), "best_fiber_type": best_ft, "best_propagation_delay_ms": round(float(best_delay or 0.0),6)}, "tables": {"fiber_delay_comparison": rows}}

def _answer_hcf_upgrade(query: str) -> Dict[str, Any]:
    ql = (query or "").lower()
    m = re.search(r"\bonly\s+(\d+)\s+links\b", ql)
    k = int(m.group(1)) if m else 2
    df = _edges_df().dropna(subset=["distance_km"]).sort_values("distance_km", ascending=False).head(k)
    links = df[["u","v","link_id","distance_km","type"]].to_dict(orient="records")
    return {"result": {"k": k, "recommended_links": links, "note": "Heuristic: upgrade the longest-distance links to HCF for maximum delay reduction (uniform traffic assumption)."}}

def _answer_min_ila_route(query: str, flags: Dict[str, bool]) -> Tuple[Dict[str, Any], Any]:
    _ensure_equipment()
    src, dst = extract_src_dst(query)
    if not src or not dst:
        return {"result": {"error": "NO_PATH"}}, None
    a, b = resolve_nodes(net.G, src, dst)

    # Dijkstra with ila_count weights
    def w(u, v, d):
        try:
            return float(d.get("ila_count", 0) or 0)
        except Exception:
            return 0.0
    try:
        path = nx.dijkstra_path(net.G, source=a, target=b, weight=w)
    except Exception:
        return {"result": {"error": "NO_PATH"}}, None

    total_ilas = 0
    for u, v in zip(path, path[1:]):
        total_ilas += int((net.G.get_edge_data(u, v) or {}).get("ila_count", 0) or 0)

    out = {"result": {"path": path, "total_ilas": int(total_ilas), "hop_count": len(path)-1, "total_distance_km": round(_path_total_distance_km(path),6)}}
    fig = None
    if flags["viz_any"] and not flags["no_plot"]:
        out["plot"] = {"title": f"Min-ILA path: {src} → {dst}", "highlight_nodes": path, "highlight_edges": _path_edges(path), "only_highlight": True, "path_only": True, "highlight_only": True, "show_edge_labels": False, "label_font_size": 7}
        _, fig = net.render_plot(out["plot"])
    return out, fig

def _answer_paths_with_ila_threshold(query: str) -> Dict[str, Any]:
    _ensure_equipment()
    src, dst = extract_src_dst(query)
    if not src or not dst:
        return {"result": {"error": "NO_PATH"}}
    a, b = resolve_nodes(net.G, src, dst)
    ql = (query or "").lower().replace("≥", ">=")
    m = re.search(r"\b(?:ilas|ila)\s*(?:>=|≥)\s*(\d+)\b", ql)
    th = int(m.group(1)) if m else 4

    max_hops = _parse_max_hops(query) or 12
    paths = []
    for p in nx.all_simple_paths(net.G, source=a, target=b, cutoff=max_hops):
        p = list(p)
        # path qualifies if any edge has ila_count >= th
        qualifies = False
        for u, v in zip(p, p[1:]):
            ila = int((net.G.get_edge_data(u, v) or {}).get("ila_count", 0) or 0)
            if ila >= th:
                qualifies = True
                break
        if qualifies:
            paths.append(p)

    return {"result": {"src": src, "dst": dst, "threshold": th, "max_hops": max_hops, "count": len(paths), "paths": paths}}

# ------------------------------ LLM fallback prompt ------------------------------
_SYSTEM_PROMPT = """You are an assistant for optical network analytics.

OUTPUT FORMAT (HARD RULE):
- Return exactly ONE valid JSON object.
- No markdown, no code fences, no extra text.
- Allowed top-level keys ONLY: result, tables, plot.
- 'plot' MUST be top-level (never nested inside result).

FAIL-INSTEAD-OF-GUESS (HARD RULE):
- If you cannot answer exactly from the provided payload, return:
  { "result": { "error": "NOT_IMPLEMENTED" } }
"""

def _llm_fallback(query: str) -> Tuple[Dict[str, Any], str, Optional[int], Optional[int], Optional[int]]:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": query}]
    raw, pt, ct, tt = call_llm(messages)
    try:
        js = _extract_first_json(raw)
    except Exception:
        js = {"result": {"error": "NOT_IMPLEMENTED"}}
    return js, raw, pt, ct, tt

# ------------------------------ Main entrypoint ------------------------------
def run(query: str) -> Dict[str, Any]:
    """Main entry point for Streamlit."""
    flags = _query_flags(query)

    plan = llm_plan(query)  # may be None
    llm_json: Dict[str, Any] = {"result": {"error": "NOT_IMPLEMENTED"}}
    fig = None
    node_df = pd.DataFrame()
    link_df = pd.DataFrame()
    qot_df = pd.DataFrame()

    def want_viz() -> bool:
        if isinstance(plan, dict) and "visualize" in plan:
            return bool(plan["visualize"])
        return bool(flags.get("viz_any", False)) and not bool(flags.get("no_plot", False))

    # ---------- LLM-plan dispatch ----------
    try:
        if isinstance(plan, dict):
            intent = plan.get("intent")

            if intent == "TOPOLOGY_BUILD":
                llm_json, fig = _answer_topology({"viz_any": want_viz(), "no_plot": not want_viz(), "distance_labels": bool(flags.get("distance_labels", False))})

            elif intent in {"SHORTEST_PATH","FEWEST_HOPS","LONGEST_PATH","ALL_PATHS"}:
                src = plan.get("src")
                dst = plan.get("dst")
                if not src or not dst:
                    src, dst = extract_src_dst(query)
                key = {
                    "SHORTEST_PATH": "shortest",
                    "FEWEST_HOPS": "fewest hops",
                    "LONGEST_PATH": "longest",
                    "ALL_PATHS": "all paths",
                }[intent]
                q2 = f"{key} path from {src} to {dst}"
                llm_json, fig, node_df, link_df = _answer_paths(q2, {"viz_any": want_viz(), "no_plot": not want_viz()})

            elif intent == "GRAPH_STATS":
                llm_json = _answer_graph_stats(query)

            elif intent == "LINK_STATS":
                llm_json, fig, link_df = _answer_link_stats(query, {"viz_any": want_viz(), "no_plot": not want_viz()})

            elif intent == "EQUIPMENT":
                llm_json, fig, node_df, link_df = _answer_equipment(query)

            elif intent == "PROP_DELAY":
                llm_json, fig = _answer_propagation_delay(query, {"viz_any": want_viz(), "no_plot": not want_viz()})

            elif intent == "FIBER_CHOICE":
                llm_json = _answer_fiber_min_delay(query)

            elif intent == "HCF_UPGRADE":
                llm_json = _answer_hcf_upgrade(query)

            elif intent == "ILA_MIN_ROUTE":
                llm_json, fig = _answer_min_ila_route(query, {"viz_any": want_viz(), "no_plot": not want_viz()})

            elif intent == "ILA_THRESH_PATHS":
                llm_json = _answer_paths_with_ila_threshold(query)
    except Exception:
        # ignore and fall back
        pass

    # ---------- Deterministic fallback ----------
    if not isinstance(llm_json, dict) or llm_json.get("result", {}).get("error") in {"NOT_IMPLEMENTED"}:
        try:
            if flags["topology"]:
                llm_json, fig = _answer_topology(flags)
            elif flags["path"]:
                llm_json, fig, node_df, link_df = _answer_paths(query, flags)
            elif flags["graph_stats"]:
                llm_json = _answer_graph_stats(query)
            elif flags["link_stats"]:
                llm_json, fig, link_df = _answer_link_stats(query, flags)
            elif flags["equip"]:
                llm_json, fig, node_df, link_df = _answer_equipment(query)
            elif flags["delay"]:
                llm_json, fig = _answer_propagation_delay(query, {"viz_any": want_viz(), "no_plot": not want_viz()})
            elif flags["fiber_choice"]:
                llm_json = _answer_fiber_min_delay(query)
            elif flags["hcf_upgrade"]:
                llm_json = _answer_hcf_upgrade(query)
            elif flags["ila_min"]:
                llm_json, fig = _answer_min_ila_route(query, {"viz_any": want_viz(), "no_plot": not want_viz()})
            elif flags["ila_thresh"]:
                llm_json = _answer_paths_with_ila_threshold(query)
        except Exception as e:
            llm_json = {"result": {"error": "EXCEPTION", "message": str(e)}}

    return {
        "llm_json": llm_json,
        "fig": fig,
        "node_df": node_df if isinstance(node_df, pd.DataFrame) else pd.DataFrame(),
        "link_df": link_df if isinstance(link_df, pd.DataFrame) else pd.DataFrame(),
        "qot_df": qot_df if isinstance(qot_df, pd.DataFrame) else pd.DataFrame(),
        "raw": json.dumps(llm_json, indent=2),
    }

