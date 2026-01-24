import pandas as pd
import math
from qot_utils import calc_qot_metrics

class EquipmentManager:
    """Handles node/link equipment deployment and optical summaries."""

    def __init__(self, graph_processor):
        self.net = graph_processor

        # Equipment parameters (professional names, no emojis)
        # Gains are representative EDFA values; NF matches your sheet (6 dB).
        self.equipment_params = {
            "ROADM": {"roadm_insertion_loss_dB": 10, "noise_figure_dB": 6},
            "TRx": {"roadm_insertion_loss_dB": 2, "noise_figure_dB": 4},

            # Amplifying elements (used on links, may be placed as PreAmp/Booster/ILA/Amp)
            "PreAmp": {"gain_dB": 20, "noise_figure_dB": 6},
            "Booster": {"gain_dB": 20, "noise_figure_dB": 6},
            "Amp": {"gain_dB": 20, "noise_figure_dB": 6},
            "ILA": {"gain_dB": 20, "noise_figure_dB": 6},
        }

    # ---------------- Node/Link Equipment Helpers ----------------
    def _add_node_equipment(self, node, label):
        eq = self.net.G.nodes[node].get("equipment_list", [])
        if not isinstance(eq, list):
            eq = list(eq) if eq else []
        if label not in eq:
            eq.append(label)
            self.net.G.nodes[node]["equipment_list"] = eq
            return True
        return False

    def _add_link_equipment(self, u, v, label):
        edge = self.net.G[u][v]
        existing = edge.get("equipment_list", [])
        if isinstance(existing, str):
            existing = [e.strip() for e in existing.split("|") if e.strip()]
        if not isinstance(existing, list):
            existing = list(existing) if existing else []
        if label not in existing:
            existing.append(label)
            edge["equipment_list"] = existing
            edge["equipment"] = " | ".join(existing)
            return True
        return False

    # ---------------- Deployment ----------------
    def deploy_equipment(self):
        """
        Deploy equipment based on node/link type and dynamic spacing rules:
        - Metro nodes get ROADM_x and TRx_x.
        - Metro links (> net.ila_spacing_km) get ILA_x, Amp_x, PreAmp_x.
        - Access links (AL) get no equipment.
        """
        G = self.net.G
        if G is None or len(G.nodes) == 0:
            return

        ila_spacing = float(getattr(self.net, "ila_spacing_km", 70.0))

        # Counters for unique equipment IDs
        roadm_counter = 1
        trx_counter = 1
        ila_counter = 1
        amp_counter = 1
        preamp_counter = 1
        booster_counter = 1

        # --- Node Equipment Deployment ---
        for node in G.nodes():
            # Metro node if it touches any ML edge
            is_metro_node = any(
                str(d.get("link_id", "")).upper().startswith("ML")
                for _, _, d in G.edges(node, data=True)
            )
            if is_metro_node:
                self._add_node_equipment(node, f"ROADM_{roadm_counter}")
                self._add_node_equipment(node, f"TRx_{trx_counter}")
                roadm_counter += 1
                trx_counter += 1

        # --- Link Equipment Deployment ---
        for u, v, data in G.edges(data=True):
            link_id = str(data.get("link_id", "")).upper()
            dist = float(data.get("weight", 0.0) or 0.0)

            # Ensure fiber_type exists for delay computations
            if not data.get("fiber_type"):
                data["fiber_type"] = "SSMF"

            # Compute and store ILA count for EVERY link (used for ILA-routing queries)
            if ila_spacing > 0:
                ila_count = max(0, int(math.ceil(dist / ila_spacing) - 1))
            else:
                ila_count = 0
            data["ila_count"] = int(ila_count)

            # Skip access links for equipment placement, but keep ila_count/fiber_type stored
            if link_id.startswith("AL"):
                continue

            # Deploy equipment on metro links when ILAs are needed
            if link_id.startswith("ML") and ila_count > 0:
                self._add_link_equipment(u, v, f"ILA_{ila_counter}")
                self._add_link_equipment(u, v, f"Amp_{amp_counter}")
                self._add_link_equipment(u, v, f"PreAmp_{preamp_counter}")
                # Optional: also add Booster if you want it in the table
                # self._add_link_equipment(u, v, f"Booster_{booster_counter}")

                ila_counter += 1
                amp_counter += 1
                preamp_counter += 1
                booster_counter += 1
        # Update optical parameters
        self.assign_equipment_optical_params()

    # ---------------- Optical Parameters ----------------
    def assign_equipment_optical_params(self):
        """Aggregate optical parameters (insertion loss, gain, noise figure) per node/link."""
        # Nodes
        for node, data in self.net.G.nodes(data=True):
            eqs = data.get("equipment_list", []) or []
            optical = {}
            for eq in eqs:
                base = eq.split("_")[0]
                params = self.equipment_params.get(base, {})
                for k, v in params.items():
                    optical[k] = optical.get(k, 0.0) + float(v)
            if optical:
                data["optical_params"] = optical

        # Links
        for u, v, data in self.net.G.edges(data=True):
            eqs = data.get("equipment_list", []) or []
            optical = {}
            for eq in eqs:
                base = eq.split("_")[0]
                params = self.equipment_params.get(base, {})
                for k, v in params.items():
                    optical[k] = optical.get(k, 0.0) + float(v)
            if optical:
                data["optical_params"] = optical

    # ---------------- Equipment Summary ----------------
    def summarize_equipment(self, path=None, visualize=False):
        """
        Summarize equipment and optical parameters for all nodes/links,
        or for a specific path (if provided).
        Returns dict with node_df, link_df, fig.
        """
        G = self.net.G
        if G is None or len(G.nodes) == 0:
            return {"node_df": pd.DataFrame(), "link_df": pd.DataFrame(), "fig": None}

        # Build node subset and link subset
        node_subset = path if path else list(G.nodes)
        link_subset = list(zip(path, path[1:])) if path else list(G.edges)

        node_rows, link_rows = [], []
        total_insertion = total_gain = total_noise = 0.0

        # Node Table
        for node in node_subset:
            d = G.nodes[node]
            eqs = d.get("equipment_list", []) or []
            optical = d.get("optical_params", {}) or {}
            insertion = optical.get("roadm_insertion_loss_dB", 0.0)
            noise = optical.get("noise_figure_dB", 0.0)

            total_insertion += insertion
            total_noise += noise

            node_rows.append({
                "Node": node,
                "Equipment": " | ".join(eqs) if eqs else "None",
                "Insertion Loss (dB)": insertion,
                "Noise Figure (dB)": noise
            })

        node_rows.append({
            "Node": "TOTAL",
            "Equipment": "",
            "Insertion Loss (dB)": total_insertion,
            "Noise Figure (dB)": total_noise
        })

        # Link Table
        # Fiber selection:
        # - If an edge has a 'fiber_type' attribute, we use it.
        # - Otherwise we fall back to net.qot_params['fiber_type'] if provided.
        default_fiber_type = None
        try:
            qp = getattr(self.net, "qot_params", None)
            if isinstance(qp, dict):
                default_fiber_type = qp.get("fiber_type")
        except Exception:
            default_fiber_type = None

        for edge in link_subset:
            u, v = (edge if len(edge) == 2 else (edge[0], edge[1]))

            d = G[u][v]
            eqs = d.get("equipment_list", []) or []
            optical = d.get("optical_params", {}) or {}
            gain = float(optical.get("gain_dB", 0.0))
            noise = float(optical.get("noise_figure_dB", 0.0))
            distance = float(d.get("weight", 0.0) or 0.0)
            total_gain += gain
            total_noise += noise

            edge_fiber_type = d.get("fiber_type") or default_fiber_type
            qot_metrics = calc_qot_metrics(distance, fiber_type=edge_fiber_type, optical_params=optical)
            attenuation = qot_metrics.get("Total Attenuation (dB)", 0.0)
            delay = qot_metrics.get("Total Propagation Delay (ms)", 0.0)

            link_rows.append({
                "Source": u,
                "Destination": v,
                "Link": f"{u} ↔ {v}",
                "Equipment": " | ".join(eqs) if eqs else "None",
                "Gain (dB)": gain,
                "Noise Figure (dB)": noise,
                "Distance (km)": distance,
                "Total Attenuation (dB)": round(attenuation, 2),
                "Total Propagation Delay (ms)": round(delay, 4)
            })

        link_rows.append({
            "Source": "",
            "Destination": "",
            "Link": "TOTAL",
            "Equipment": "",
            "Gain (dB)": total_gain,
            "Noise Figure (dB)": total_noise,
            "Distance (km)": 0,
            "Total Attenuation (dB)": 0,
            "Total Propagation Delay (ms)": 0
        })

        node_df = pd.DataFrame(node_rows)
        link_df = pd.DataFrame(link_rows)

        # Optional visualization
        fig = None
        if visualize:
            try:
                viz = self.net._visualize_graph(highlight_edges=link_subset, title="Network with Equipment & QoT")
                if isinstance(viz, tuple):
                    import matplotlib.figure as mfig
                    for item in viz:
                        if isinstance(item, mfig.Figure):
                            fig = item
                            break
            except Exception as e:
                print(f"[Warning] Visualization failed in summarize_equipment: {e}")
                fig = None

        return {"node_df": node_df, "link_df": link_df, "fig": fig}