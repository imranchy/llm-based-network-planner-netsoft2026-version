import math
import pandas as pd
import networkx as nx

from qot_utils import calc_qot_metrics, OPTICAL_CONSTANTS, calc_ase_noise, dBm_to_mW
from equipment_manager import EquipmentManager

class QoTAnalyzer:
    """Compute QoT metrics for paths in a network graph, fully equipment-aware and Streamlit-ready."""

    def __init__(self, graph_processor):
        self.graph_proc = graph_processor
        self.G = graph_processor.G
        # Use EquipmentManager from the same graph (do not redeploy on init)
        self.eqp_mngr = EquipmentManager(graph_processor)
        self._cache = {}
        self._link_df_cache = pd.DataFrame()  # cache link QoT DataFrame

    # ---------------- Node normalization ----------------
    def _normalize_node(self, node):
        """Normalize node input to match graph nodes (case-insensitive)."""
        if not isinstance(node, str):
            return node
        node = node.strip("'\"")
        node_map = {n.upper(): n for n in self.G.nodes}
        return node_map.get(node.upper(), None)

    # ---------------- Cached link dataframe ----------------
    def _get_link_df(self):
        """Retrieve or compute cached link QoT DataFrame. Always returns a DataFrame."""
        if self._link_df_cache.empty:
            try:
                summary = self.eqp_mngr.summarize_equipment(visualize=False)
                if isinstance(summary, dict):
                    link_df = summary.get("link_df", pd.DataFrame())
                elif isinstance(summary, tuple) and len(summary) >= 2:
                    _, link_df, _ = summary
                else:
                    link_df = pd.DataFrame()
                if link_df is None or not isinstance(link_df, pd.DataFrame):
                    link_df = pd.DataFrame()
                self._link_df_cache = link_df.copy()
            except Exception as e:
                print(f"[Warning] Failed to get link_df from equipment manager: {e}")
                self._link_df_cache = pd.DataFrame()
        return self._link_df_cache

    # ---------------- Path Analysis ----------------
    def analyze_path(self, src, dst, max_paths=None, visualize=False):
        """
        Compute equipment-aware QoT metrics between SRC and DST.
        Returns a dict with DataFrames and figures suitable for Streamlit.
        """
        src, dst = self._normalize_node(src), self._normalize_node(dst)
        if src is None or dst is None:
            return {"error": f"Invalid node(s): {src}, {dst}"}

        cache_key = (src, dst)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # --- Compute all simple paths ---
        try:
            all_paths = list(nx.all_simple_paths(self.G, source=src, target=dst))
        except nx.NetworkXNoPath:
            return {"error": f"No valid path between {src} and {dst}"}
        except Exception as e:
            return {"error": f"Path computation failed: {e}"}

        if not all_paths:
            return {"error": f"No paths found between {src} and {dst}"}

        if max_paths is not None:
            all_paths = all_paths[:max_paths]

        # --- Safe Link DataFrame retrieval ---
        link_df = self._get_link_df()
        if not isinstance(link_df, pd.DataFrame):
            print(f"[Warning] Expected link_df as DataFrame, got {type(link_df).__name__}")
            link_df = pd.DataFrame()

        if link_df.empty:
            return {
                "error": "No link data available — please deploy equipment first using `Deploy network equipment`."
            }

        # --- Compute metrics per path ---
        table_rows = []
        for idx, path in enumerate(all_paths, start=1):
            total_distance = total_att = total_gain = total_noise = total_delay = 0.0

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]

                # Prefer Link column, accept both "↔" and "-" separators
                row = pd.DataFrame()
                if "Link" in link_df.columns:
                    row = link_df.loc[
                        (link_df["Link"] == f"{u} ↔ {v}") | (link_df["Link"] == f"{v} ↔ {u}") |
                        (link_df["Link"] == f"{u} - {v}") | (link_df["Link"] == f"{v} - {u}")
                    ]
                elif {"Source", "Destination"}.issubset(link_df.columns):
                    row = link_df.loc[
                        ((link_df["Source"] == u) & (link_df["Destination"] == v)) |
                        ((link_df["Source"] == v) & (link_df["Destination"] == u))
                    ]

                if isinstance(row, pd.DataFrame) and not row.empty:
                    dist = float(row["Distance (km)"].values[0]) if "Distance (km)" in row.columns else float(self.G[u][v].get("weight", 0))
                    attenuation = float(row["Total Attenuation (dB)"].values[0]) if "Total Attenuation (dB)" in row.columns else 0.0
                    delay = float(row["Total Propagation Delay (ms)"].values[0]) if "Total Propagation Delay (ms)" in row.columns else 0.0
                    gain = float(row["Gain (dB)"].values[0]) if "Gain (dB)" in row.columns else 0.0
                    noise = float(row["Noise Figure (dB)"].values[0]) if "Noise Figure (dB)" in row.columns else 0.0
                else:
                    # Fall back to raw graph data
                    dist = float(self.G[u][v].get("weight", 0))
                    optical = self.G[u][v].get("optical_params", {}) or {}
                    qot = calc_qot_metrics(dist, optical_params=optical)
                    attenuation = qot.get("Total Attenuation (dB)", 0)
                    delay = qot.get("Total Propagation Delay (ms)", 0)
                    gain = qot.get("Gain (dB)", 0)
                    noise = qot.get("Noise Figure (dB)", 0)

                total_distance += dist
                total_att += attenuation
                total_delay += delay
                total_gain += gain
                total_noise += noise

            table_rows.append({
                "Path No #": idx,
                "Nodes": path,
                "Hop Count": len(path) - 1,
                "Total Distance (km)": round(total_distance, 2),
                "Total Gain (dB)": round(total_gain, 2),
                "Total Noise Figure (dB)": round(total_noise, 2),
                "Total Attenuation (dB)": round(total_att, 2),
                "Total Propagation Delay (ms)": round(total_delay, 4)
            })

        df = pd.DataFrame(table_rows)
        if df.empty:
            return {"error": "No valid paths computed."}

        # --- Shortest and longest paths ---
        shortest = df.loc[df["Total Distance (km)"].idxmin()]
        longest = df.loc[df["Total Distance (km)"].idxmax()]

        # --- Visualizations ---
        figs = {}
        if visualize:
            try:
                viz_short = self.graph_proc._visualize_path(shortest["Nodes"], title=f"Shortest Path: {src} → {dst}")
                figs["shortest_path"] = viz_short[1] if isinstance(viz_short, tuple) else viz_short
            except Exception as e:
                print(f"[Warning] Visualization (shortest) failed: {e}")

            try:
                viz_long = self.graph_proc._visualize_path(longest["Nodes"], title=f"Longest Path: {src} → {dst}")
                figs["longest_path"] = viz_long[1] if isinstance(viz_long, tuple) else viz_long
            except Exception as e:
                print(f"[Warning] Visualization (longest) failed: {e}")

        # --- Pack results ---
        result = {
            "source": src,
            "destination": dst,
            "path_count": len(df),
            "shortest_path": shortest.to_frame().T,
            "longest_path": longest.to_frame().T,
            "full_table": df,
            "figures": figs
        }

        self._cache[cache_key] = result
        return result

    # ---------------- QoT Detailed Metrics ----------------
    def calculate_path_qot(self, src, dst, visualize=False):
        """
        Compute physics-based QoT metrics (OSNR via ASE accumulation, GSNR, Q, BER, ROP, Latency)
        between SRC and DST using equipment-aware path data.
        """
        result = self.analyze_path(src, dst, visualize=visualize)
        if "error" in result:
            return result

        df = result["full_table"]
        if df.empty:
            return {"error": "No path data available to compute QoT metrics."}

        path_data = df.loc[df["Total Distance (km)"].idxmin()]
        total_distance = float(path_data["Total Distance (km)"])
        total_att_dB = float(path_data["Total Attenuation (dB)"])
        total_gain_dB = float(path_data["Total Gain (dB)"])
        total_noise_dB = float(path_data["Total Noise Figure (dB)"])
        latency_ms = float(path_data["Total Propagation Delay (ms)"])

        # ------------------ Count inline amplifiers dynamically ------------------
        ila_spacing = float(getattr(self.graph_proc, "ila_spacing_km", 70.0))
        num_amps = 0
        shortest_path_nodes = list(path_data["Nodes"])
        for i in range(len(shortest_path_nodes) - 1):
            u, v = shortest_path_nodes[i], shortest_path_nodes[i + 1]
            edge = self.G[u][v]
            link_id = str(edge.get("link_id", "")).upper()
            dist = float(edge.get("weight", 0.0) or 0.0)
            if link_id.startswith("ML"):
                num_amps += max(0, math.ceil(dist / ila_spacing) - 1)

        # ------------------ Signal power at path output ------------------
        Ptx_mW = dBm_to_mW(OPTICAL_CONSTANTS["tx_power_dBm"])
        Pout_mW = Ptx_mW * (10 ** ((total_gain_dB - total_att_dB) / 10.0))  # end-of-path power in mW
        Pout_W = Pout_mW / 1000.0

        # ------------------ ASE noise accumulation ----------------------
        P_ase_W_total = 0.0
        if num_amps <= 0:
            num_amps = 1  # at least terminal amplification/noise contribution
        for _ in range(num_amps):
            P_ase_W_total += calc_ase_noise(gain_dB=20.0, nf_dB=OPTICAL_CONSTANTS["amp_noise_figure_dB"])

        # ------------------ OSNR / GSNR ----------------------
        OSNR_linear = (Pout_W) / max(P_ase_W_total, 1e-24)
        osnr_dB = 10.0 * math.log10(OSNR_linear)
        gsnr_dB = osnr_dB - total_noise_dB  # simple NF penalty

        # ------------------ Q-factor & BER (AWGN approx) -----
        Q = (10 ** (gsnr_dB / 20.0)) / math.sqrt(2.0)
        ber = 0.5 * math.erfc(Q / math.sqrt(2.0))

        # ------------------ Received Optical Power (dBm) -----
        rop_dBm = OPTICAL_CONSTANTS["tx_power_dBm"] - total_att_dB + total_gain_dB

        qot_summary = pd.DataFrame([{
            "Source": src,
            "Destination": dst,
            "Distance (km)": round(total_distance, 2),
            "Inline Amplifiers (count)": int(num_amps),
            "Total Attenuation (dB)": round(total_att_dB, 2),
            "Total Gain (dB)": round(total_gain_dB, 2),
            "OSNR (dB)": round(osnr_dB, 2),
            "GSNR (dB)": round(gsnr_dB, 2),
            "Q-factor": round(Q, 3),
            "BER": f"{ber:.2e}",
            "ROP (dBm)": round(rop_dBm, 2),
            "Latency (ms)": round(latency_ms, 4),
        }])

        result.update({
            "advanced_qot": qot_summary,
            "metrics": {
                "osnr_dB": osnr_dB,
                "gsnr_dB": gsnr_dB,
                "q_factor": Q,
                "ber": ber,
                "rop_dBm": rop_dBm,
                "latency_ms": latency_ms,
                "num_amps": int(num_amps),
            }
        })

        return result
