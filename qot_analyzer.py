"""qot_analyzer.py

Minimal QoT analyzer used by NetworkGraph.query_qot_metric().

The full optical QoT model is out of scope for this demo; we implement just enough
to support propagation delay / latency queries in a deterministic way.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

import networkx as nx

from qot_utils import calc_qot_metrics


class QoTAnalyzer:
    def __init__(self, net: Any):
        self.net = net  # expects an object with .G (networkx graph)

    def _path_distance_km(self, path: List[str]) -> float:
        total = 0.0
        for u, v in zip(path, path[1:]):
            d = (self.net.G.get_edge_data(u, v) or {})
            w = d.get("weight", 0.0)
            try:
                total += float(w)
            except Exception:
                pass
        return float(total)

    def calculate_path_qot(self, src: str, dst: str, visualize: bool = False, fiber_type: str = "SSMF") -> Dict[str, Any]:
        try:
            path = nx.shortest_path(self.net.G, source=src, target=dst, weight="weight")
        except Exception as e:
            return {"error": "NO_PATH", "message": str(e)}

        total_km = self._path_distance_km(path)
        metrics = calc_qot_metrics(total_km, fiber_type=fiber_type, optical_params=None)

        # Provide a structure compatible with graph_processor.query_qot_metric
        return {
            "path": path,
            "metrics": {
                "latency_ms": float(metrics.get("Total Propagation Delay (ms)", 0.0) or 0.0),
                # placeholders for compatibility
                "ber": None,
                "osnr_dB": None,
                "gsnr_dB": None,
                "rop_dBm": None,
            },
            "advanced_qot": {"fiber_type": str(fiber_type).upper(), "distance_km": float(total_km)},
        }
