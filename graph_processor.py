"""
graph_processor.py
------------------
Network topology analysis and visualization tool.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
from qot_analyzer import QoTAnalyzer
from graph_utils import interactive_plot, resolve_nodes
from qot_utils import calc_qot_metrics
from ui_utils import progress_update, timed
from io_utils import save_dataframe
from equipment_manager import EquipmentManager


class NetworkGraph:
    def __init__(self, dist_file: str, city_file: str):
        self.dist_df = pd.read_csv(dist_file)
        self.city_df = pd.read_csv(city_file)

        self.G = nx.Graph()
        self.pos = None

        # Dynamic amplifier/ILA spacing (km)
        self.ila_spacing_km = 70.0

        # QoT parameters for display/edit (optional YAML edits)
        self.qot_params = {
            "Fiber attenuation (dB/km)": 0.2,
            "ROADM insertion loss (dB)": 10,
            "Amplifier NF (dB)": 6,
            "Tx power (dBm)": 0,
            "Symbol rate (GBd)": 64,
            "Roll-off (beta)": 0.1,
            "In-band bandwidth Bn (GHz)": 70.4,
            "OSNR reference bandwidth (GHz)": 12.5,
            "Wavelength (nm)": 1550,
            "Planck constant (J*s)": 6.62607e-34,
            "Optical frequency (Hz)": 1.93414e14
        }

        self._build_graph()
        self._cache_positions()

        self.qot_analyzer = QoTAnalyzer(self)

    # ----- Setters for LLM/YAML -----
    def set_ila_spacing(self, spacing_km: float):
        try:
            spacing = float(spacing_km)
            if spacing > 0:
                self.ila_spacing_km = spacing
        except Exception:
            pass

    def set_qot_param(self, key: str, value):
        if key in self.qot_params:
            try:
                self.qot_params[key] = float(value)
            except Exception:
                self.qot_params[key] = value

    # ----- Build graph from CSVs -----
    def _build_graph(self):
        for _, row in self.dist_df.iterrows():
            self.G.add_edge(
                row["Node A"],
                row["Node Z"],
                weight=float(row["Distance (km)"]),
                link_id=row.get("Link_id")
            )

    # ----- Cache positions -----
    def _cache_positions(self, layout=None):
        if "Latitude" in self.city_df.columns and "Longitude" in self.city_df.columns:
            self.pos = {
                row["City"]: (row["Longitude"], row["Latitude"])
                for _, row in self.city_df.iterrows()
            }
        else:
            self.pos = layout or nx.spring_layout(self.G)

    # ==========================
    # QoT Query Interface
    # ==========================
    def query_qot_metric(self, src, dst, metric="BER", visualize=False):
        try:
            result = self.qot_analyzer.calculate_path_qot(src, dst, visualize=visualize)
            if "error" in result:
                return result
            metrics = result.get("metrics", {})
            advanced_qot = result.get("advanced_qot", None)
            metric_key = metric.strip().upper()
            mapping = {
                "BER": "ber",
                "OSNR": "osnr_dB",
                "GSNR": "gsnr_dB",
                "ROP": "rop_dBm",
                "LATENCY": "latency_ms"
            }
            if metric_key in mapping and mapping[metric_key] in metrics:
                return {
                    "metric": metric_key,
                    "value": metrics[mapping[metric_key]],
                    "unit": {
                        "BER": "",
                        "OSNR": "dB",
                        "GSNR": "dB",
                        "ROP": "dBm",
                        "LATENCY": "ms"
                    }.get(metric_key, ""),
                    "details": advanced_qot
                }
            else:
                return {
                    "metric": metric_key,
                    "value": None,
                    "details": advanced_qot,
                    "warning": f"Metric '{metric}' not recognized. Available: {list(mapping.keys())}"
                }
        except Exception as e:
            print(f"[Error] QoT metric query failed: {e}")
            return {"error": str(e)}

    # ----- Synthetic topology generation -----
    def generate_topology(
        self,
        network_type="metro",
        num_metro=5,
        num_access_per_metro=2,
        avg_degree=2,
        min_dist=1.0,
        max_dist=10.0,
        seed=42,
        **kwargs
    ):
        random.seed(seed)
        ml_counter, al_counter = 1, 1
        G_new = nx.Graph()

        if network_type.lower() == "metro":
            G_new = nx.connected_watts_strogatz_graph(
                num_metro, k=int(avg_degree), p=0.3, seed=seed
            )
            metro_nodes = [f"M{i+1}" for i in range(num_metro)]
            G_new = nx.relabel_nodes(G_new, dict(zip(G_new.nodes(), metro_nodes)))
            for u, v in G_new.edges():
                dist = round(random.uniform(min_dist, max_dist), 1)
                G_new[u][v].update(
                    {"weight": dist, "link_id": f"ML{ml_counter}", "type": "metro"}
                )
                ml_counter += 1

        elif network_type.lower() in ["converged", "access"]:
            G_new = nx.connected_watts_strogatz_graph(
                num_metro, k=int(avg_degree), p=0.3, seed=seed
            )
            metro_nodes = [f"M{i+1}" for i in range(num_metro)]
            G_new = nx.relabel_nodes(G_new, dict(zip(G_new.nodes(), metro_nodes)))
            for u, v in G_new.edges():
                dist = round(random.uniform(min_dist, max_dist), 1)
                G_new[u][v].update(
                    {"weight": dist, "link_id": f"ML{ml_counter}", "type": "metro"}
                )
                ml_counter += 1

            access_counter = 1
            for metro in metro_nodes:
                for _ in range(num_access_per_metro):
                    access_node = f"A{access_counter}"
                    access_counter += 1
                    G_new.add_node(access_node)
                    dist = round(random.uniform(min_dist, max_dist), 1)
                    G_new.add_edge(
                        metro,
                        access_node,
                        weight=dist,
                        link_id=f"AL{al_counter}",
                        type="access"
                    )
                    al_counter += 1

        self.G = G_new
        self.pos = nx.spring_layout(G_new, seed=seed)
        return G_new

    # ----- Build/generate + deploy + summarize -----
    def build_topology(
        self,
        visualize=True,
        use_csv=False,
        use_random=False,
        network_type="metro",
        **kwargs
    ):
        """
        Build or generate network topology, deploy equipment,
        summarize deployment, and return (fig, node_df, link_df, qot_df).
        """
        try:
            # Build or generate
            if use_csv:
                # Clear then rebuild from CSV to avoid duplication across runs
                self.G.clear()
                self._build_graph()
                self._cache_positions()
            else:
                if use_random:
                    kwargs.setdefault("network_type", "metro")
                if "network_type" in kwargs:
                    kwargs.pop("network_type")
                self.generate_topology(network_type=network_type, **kwargs)

            # Choose a sensible default ILA spacing for generated graphs
            if not use_csv:
                max_dist = kwargs.get("max_dist", 60.0)
                try:
                    max_dist = float(max_dist)
                except Exception:
                    max_dist = 60.0
                if str(network_type).lower() == "metro":
                    self.ila_spacing_km = max(40.0, min(max_dist, 80.0))
                else:
                    self.ila_spacing_km = 70.0

            # Deploy and summarize equipment
            eqp_mngr = EquipmentManager(self)
            eqp_mngr.deploy_equipment()

            summary = eqp_mngr.summarize_equipment(visualize=False)
            node_df = summary["node_df"]
            link_df = summary["link_df"]

            # QoT parameter table for UI
            qot_df = pd.DataFrame(
                [{"Parameter": k, "Value": v} for k, v in self.qot_params.items()]
            )

            # Visualization
            fig = None
            if visualize:
                _, fig = self._visualize_graph(
                    title="Network Topology with Deployed Equipment"
                )

            return fig, node_df, link_df, qot_df

        except Exception as e:
            print(f"❌ Error during topology build: {e}")
            return None, None, None, None

    # ==========================
    # Path / Stats / Visualization helpers
    # ==========================
    @timed()
    def query_network_path(self, src, dst, mode="shortest", visualize=True, max_hops=None):
        """
        Query paths between SRC and DST.
        Returns (fig, node_df, link_df) for consistent handling in the LLM agent.
        """
        fig = None
        src, dst = resolve_nodes(self.G, src, dst)
        if src is None or dst is None:
            return None, pd.DataFrame(), pd.DataFrame()

        try:
            all_paths = list(
                nx.all_simple_paths(self.G, source=src, target=dst, cutoff=max_hops)
            )
        except nx.NetworkXNoPath:
            return None, pd.DataFrame(), pd.DataFrame()

        if not all_paths:
            return None, pd.DataFrame(), pd.DataFrame()

        # Compute metrics
        path_data = []
        for idx, path in enumerate(all_paths, start=1):
            total_distance = sum(
                self.G[path[i]][path[i + 1]].get("weight", 1.0)
                for i in range(len(path) - 1)
            )
            path_data.append({
                "Path #": idx,
                "Nodes": path,
                "Total Distance (km)": round(total_distance, 2),
                "Hop Count": len(path) - 1
            })

        node_df = pd.DataFrame(path_data)
        link_df = pd.DataFrame()  # Optional: populate later if needed

        if mode in ["shortest", "longest"]:
            best_idx = (
                node_df["Total Distance (km)"].idxmin()
                if mode == "shortest"
                else node_df["Total Distance (km)"].idxmax()
            )
            best_path = all_paths[best_idx]
            if visualize:
                _, fig = self._visualize_path(
                    best_path,
                    title=f"{mode.title()} Path: {src} → {dst}",
                    highlight_color="green" if mode == "shortest" else "red"
                )
            node_df = node_df.loc[[best_idx]]

        elif mode == "all":
            if visualize:
                highlight_edges = [
                    edge for path in all_paths for edge in zip(path, path[1:])
                ]
                _, fig = self._visualize_graph(
                    highlight_edges=highlight_edges,
                    title=f"All Paths: {src} → {dst}",
                    highlight_color="blue"
                )

        elif mode == "metric":
            metrics = {
                "Shortest Distance": node_df['Total Distance (km)'].min(),
                "Minimum Hop Count": node_df['Hop Count'].min()
            }
            node_df = pd.DataFrame([metrics])

        else:
            node_df = pd.DataFrame([{"Error": f"Unknown mode '{mode}'"}])

        if "Nodes" in node_df.columns:
            node_df["Nodes"] = node_df["Nodes"].apply(
                lambda x: list(x) if isinstance(x, (list, tuple)) else str(x).split(" → ")
            )

        return fig, node_df, link_df

    # ----- Shortest/Longest Path by Link Type -----
    @timed()
    def shortest_path_in_links(self, src, dst, link_type="ML", visualize=True):
        SG = self.subgraph_by_type(link_type)
        fig = None

        src, dst = resolve_nodes(SG, src, dst)
        if src is None or dst is None:
            return pd.DataFrame(columns=["Nodes", "Distance (km)"]), fig

        all_paths = list(nx.all_simple_paths(SG, src, dst))
        if not all_paths:
            return pd.DataFrame(columns=["Nodes", "Distance (km)"]), fig

        path_distances = [
            (path, sum(SG[path[i]][path[i + 1]].get("weight", 1.0) for i in range(len(path) - 1)))
            for path in all_paths
        ]
        shortest_path, min_dist = min(path_distances, key=lambda x: x[1])

        if visualize:
            _, fig = self._visualize_path(
                shortest_path,
                title=f"Shortest {link_type} Path: {src} → {dst}"
            )

        df = pd.DataFrame([{"Nodes": shortest_path, "Distance (km)": min_dist}])
        return df, fig

    @timed()
    def longest_path_in_links(self, src, dst, link_type="ML", visualize=True):
        SG = self.subgraph_by_type(link_type)
        fig = None

        src, dst = resolve_nodes(SG, src, dst)
        if src is None or dst is None:
            return pd.DataFrame(columns=["Nodes", "Distance (km)"]), fig

        all_paths = list(nx.all_simple_paths(SG, src, dst))
        if not all_paths:
            return pd.DataFrame(columns=["Nodes", "Distance (km)"]), fig

        path_distances = [
            (path, sum(SG[path[i]][path[i + 1]].get("weight", 1.0) for i in range(len(path) - 1)))
            for path in all_paths
        ]
        longest_path, max_dist = max(path_distances, key=lambda x: x[1])

        if visualize:
            _, fig = self._visualize_path(
                longest_path,
                title=f"Longest {link_type} Path: {src} → {dst}"
            )

        df = pd.DataFrame([{"Nodes": longest_path, "Distance (km)": max_dist}])
        return df, fig

    # ----- Average Node Degree -----
    @timed()
    def average_node_degree(self, visualize=True):
        fig = None
        degrees = dict(self.G.degree())
        avg_deg = sum(degrees.values()) / len(degrees) if degrees else 0
        min_deg = min(degrees.values()) if degrees else 0
        max_deg = max(degrees.values()) if degrees else 0

        df = pd.DataFrame({
            "Metric": ["Average Degree", "Minimum Degree", "Maximum Degree"],
            "Value": [round(avg_deg, 2), min_deg, max_deg]
        })

        return df, fig

    # ----- Average Distance Link -----
    @timed()
    def average_distance_link(self, link_type="ML", visualize=True):
        fig = None
        SG = self.subgraph_by_type(link_type)
        distances = [d.get("weight", 0) for _, _, d in SG.edges(data=True)]
        avg = sum(distances) / len(distances) if distances else 0

        df = pd.DataFrame([{
            "Link Type": link_type,
            "Average Distance (km)": round(avg, 2)
        }])

        return df, fig

    # ----- Min/Max Link Distance -----
    @timed()
    def min_max_link_distance(self, link_type="ML", find_max=False, visualize=True):
        fig = None
        SG = self.subgraph_by_type(link_type)
        edges = [(u, v, d.get("weight", 0)) for u, v, d in SG.edges(data=True)]

        if not edges:
            df = pd.DataFrame([{"Message": f"No {link_type} links found."}])
            return df, fig

        u, v, dist = max(edges, key=lambda x: x[2]) if find_max else min(edges, key=lambda x: x[2])
        color = "red" if find_max else "green"

        df = pd.DataFrame([{
            "Link Type": link_type,
            "Node A": u,
            "Node B": v,
            "Distance (km)": round(dist, 2),
            "Min/Max": "Max" if find_max else "Min"
        }])

        if visualize:
            fig = self._visualize_graph(
                highlight_edges=[(u, v)],
                title=f"{link_type} Link {'Maximum' if find_max else 'Minimum'} Distance",
                highlight_color=color
            )

        return df, fig

    # ----- Remove ML/AL links and visualize -----
    @timed()
    def remove_ml_al_links_and_visualize(self, visualize=True):
        to_remove = [(u, v) for u, v, d in self.G.edges(data=True)
                     if d.get("link_id", "").upper().startswith(("ML", "AL"))]

        self.G.remove_edges_from(to_remove)
        removed_count = len(to_remove)

        df_removed = pd.DataFrame(to_remove, columns=["Node A", "Node B"]) if removed_count else pd.DataFrame()

        fig = None
        if visualize:
            fig = self._visualize_graph(title="Network after removing ML/AL links")

        return {"removed_links_count": removed_count, "removed_links": df_removed}, fig

    # ----- Drop specific links (parameterized) -----
    @timed()
    def drop_links_and_visualize(self, links_to_remove=None, visualize=True):
        removed_links = []

        if links_to_remove:
            for u, v in links_to_remove:
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)
                    removed_links.append((u, v))

        df_removed = pd.DataFrame(removed_links, columns=["Node A", "Node B"]) if removed_links else pd.DataFrame()

        fig = None
        if visualize:
            fig = self._visualize_graph(title="Network after dropping specified links")

        return {"removed_links": df_removed, "total_removed": len(removed_links)}, fig

    # ----- Visualize by Link Type -----
    @timed()
    def visualize_topology_by_type(self, link_type="ML", visualize=True):
        SG = self.subgraph_by_type(link_type)
        info = {
            "link_type": link_type,
            "edges_count": SG.number_of_edges(),
            "nodes_count": SG.number_of_nodes()
        }

        fig = None
        if visualize and SG.number_of_edges() > 0:
            fig = self._visualize_graph(
                highlight_edges=list(SG.edges()),
                title=f"{link_type} Network Topology"
            )

        return info, fig

    # ----- Metro connected to access nodes -----
    @timed()
    def metro_connected_to_access_nodes(self, visualize=True):
        SG_metro = self.subgraph_by_type("ML")
        SG_access = self.subgraph_by_type("AL")
        access_nodes = set(SG_access.nodes())

        metro_connections = [(u, v) for u, v in SG_metro.edges() if u in access_nodes or v in access_nodes]

        df_connections = pd.DataFrame(metro_connections, columns=["Node A", "Node B"]) if metro_connections else pd.DataFrame()

        fig = None
        if visualize and not df_connections.empty:
            fig = self._visualize_graph(
                highlight_edges=metro_connections,
                title="Metro Links Connected to Access Nodes"
            )

        return {"metro_to_access_links": df_connections, "total_links": len(metro_connections)}, fig

    # ----- Links Info -----
    @timed()
    def links_info(self, link_type="ML", visualize=True):
        fig = None
        link_type = link_type.upper()
        if link_type not in {"ML", "AL"}:
            raise ValueError(f"Invalid link_type '{link_type}'. Use 'ML' or 'AL'.")

        SG = self.subgraph_by_type(link_type)
        edges_data = [(u, v, d.get("weight", 0), d.get("link_id", "-")) for u, v, d in SG.edges(data=True)]
        df = pd.DataFrame(edges_data, columns=["Node A", "Node B", "Distance (km)", "Link ID"])

        if visualize and not df.empty:
            fig = self._visualize_graph(
                highlight_edges=list(SG.edges()),
                title=f"{link_type} Links ({len(df)})"
            )

        return df, fig

    # ----- Links under distance -----
    @timed()
    def links_under_distance(self, link_type="ML", distance=0, visualize=True):
        fig = None
        link_type = link_type.upper()
        if link_type not in {"ML", "AL"}:
            raise ValueError(f"Invalid link_type '{link_type}'. Use 'ML' or 'AL'.")

        try:
            distance = float(distance)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid distance value: {distance}")

        SG = self.subgraph_by_type(link_type)
        filtered_edges = [
            (u, v, d.get("weight", 0), d.get("link_id", "-"))
            for u, v, d in SG.edges(data=True) if d.get("weight", 0) <= distance
        ]
        df = pd.DataFrame(filtered_edges, columns=["Node A", "Node B", "Distance (km)", "Link ID"])

        if visualize and not df.empty:
            highlight_edges = [(u, v) for u, v, _, _ in filtered_edges]
            fig = self._visualize_graph(
                highlight_edges=highlight_edges,
                title=f"{link_type} Links ≤ {distance} km"
            )

        return df, fig

    # ----- Subgraph by Type -----
    def subgraph_by_type(self, link_type="ML"):
        SG = nx.Graph()
        edges = [
            (u, v, d) for u, v, d in self.G.edges(data=True)
            if d.get("link_id", "").upper().startswith(link_type.upper())
        ]
        SG.add_nodes_from(self.G.nodes)
        SG.add_edges_from(edges)
        return SG

    # ----- Interactive Modify Network -----
    @timed()
    def modify_network(self, add_edges=None, remove_edges=None, visualize=True):
        """
        Add or remove edges from the network in a parameterized and safe way.
        add_edges/remove_edges:
            add_edges: list of tuples [(NodeA, NodeB, weight, link_id), ...]
            remove_edges: list of tuples [(NodeA, NodeB), ...]
        Returns:
            dict of applied changes and optional visualization figure.
        """
        fig = None
        changes = []

        # Remove Edges
        if remove_edges:
            for u, v in remove_edges:
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)
                    changes.append({"action": "remove", "nodes": (u, v)})
                else:
                    changes.append({"action": "remove_failed", "nodes": (u, v), "reason": "edge_not_found"})

        # Add Edges
        if add_edges:
            for edge in add_edges:
                if len(edge) not in [3, 4]:
                    changes.append({"action": "add_failed", "edge": edge, "reason": "invalid_format"})
                    continue

                u, v, weight = edge[:3]
                link_id = edge[3] if len(edge) == 4 else f"LINK_{u}_{v}"

                try:
                    weight = float(weight)
                except (TypeError, ValueError):
                    changes.append({"action": "add_failed", "edge": (u, v), "reason": f"invalid_weight: {weight}"})
                    continue

                self.G.add_edge(u, v, weight=weight, link_id=link_id)
                changes.append({
                    "action": "add",
                    "nodes": (u, v),
                    "weight": round(weight, 2),
                    "link_id": link_id
                })

        # Update Layout if graph changed
        if add_edges or remove_edges:
            self.pos = nx.spring_layout(self.G, seed=42)

        # Visualization
        if visualize and changes:
            fig = self._visualize_graph(
                title="Network After Modification",
                highlight_edges=[change["nodes"] for change in changes if "nodes" in change]
            )

        return {"changes": changes, "total_changes": len(changes)}, fig

    # ----- Visualization helpers -----
    @interactive_plot
    def _visualize_graph(self, highlight_edges=None, highlight_nodes=None, title="Network Graph", highlight_color="red"):
        fig, ax = plt.subplots(figsize=(12, 8))

        nx.draw(self.G, 
                self.pos, 
                with_labels=True, 
                node_color="lightblue",
                font_size=8, 
                node_size=500, 
                ax=ax)
        nx.draw_networkx_edges(self.G, self.pos, edge_color="gray", width=1.0, ax=ax)

        # Optional highlights
        if highlight_edges:
            nx.draw_networkx_edges(
                self.G,
                self.pos,
                edgelist=highlight_edges,
                edge_color=highlight_color,
                width=2.5,
                ax=ax,
            )

        if highlight_nodes:
            nx.draw_networkx_nodes(
                self.G,
                self.pos,
                nodelist=highlight_nodes,
                node_color="orange",
                node_size=850,
                ax=ax,
            )

        edge_labels = {
            (u, v): f"{d.get('link_id', '')} ({d.get('weight', '?')} km)"
            for u, v, d in self.G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=8, ax=ax)

        ax.set_title(title)
        ax.axis("off")

        info = {
            "title": title,
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "highlighted_edges_count": len(highlight_edges) if highlight_edges else 0
        }

        return info, fig

    # ==========================
    # Plot spec renderer (LLM-first)
    # ==========================
    @interactive_plot
    def _visualize_path(self, path, title="Network Path", highlight_color="red"):
        import matplotlib.pyplot as plt
        import networkx as nx

        fig, ax = plt.subplots(figsize=(12, 8))

        nx.draw(self.G, self.pos, with_labels=True, node_color="lightblue", node_size=500, ax=ax)
        nx.draw_networkx_edges(self.G, self.pos, edge_color="gray", width=1.0, ax=ax)

        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(
            self.G, self.pos, edgelist=path_edges, edge_color=highlight_color, width=3.0, ax=ax
        )
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=path, node_color="orange", node_size=800, ax=ax)

        edge_labels = {
            (u, v): f"{self.G[u][v].get('link_id', '')} ({self.G[u][v].get('weight', '?')} km)"
            for u, v in path_edges
        }
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=8, ax=ax)

        ax.set_title(title)
        ax.axis("off")

        info = {
            "title": title,
            "path_nodes": path,
            "path_edges_count": len(path_edges),
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges()
        }

        return info, fig

    # ==========================
    # LLM-first helpers
    # ==========================
    def topology_snapshot(self) -> dict:
        """Return a JSON-friendly snapshot of the current topology.

        This is the ONLY graph structure representation sent to the LLM.
        All graph analytics (shortest path, degrees, hop count, etc.) must be
        computed by the model from this snapshot.
        """
        nodes = []
        for n in self.G.nodes():
            nodes.append({"id": str(n)})

        edges = []
        for u, v, d in self.G.edges(data=True):
            edges.append(
                {
                    "u": str(u),
                    "v": str(v),
                    "distance_km": float(d.get("weight")) if d.get("weight") is not None else None,
                    "link_id": d.get("link_id"),
                    "type": d.get("type"),
                }
            )

        return {
            "directed": False,
            "nodes": nodes,
            "edges": edges,
        }

    def apply_graph_edits(self, edits: dict) -> dict:
        """Apply simple topology edits requested by the LLM.

        Supported keys:
          - add_edges: list of {u,v,distance_km,link_id,type}
          - remove_edges: list of {u,v}

        Returns a small summary dict.
        """
        applied = {"added": 0, "removed": 0, "errors": []}
        if not isinstance(edits, dict):
            return applied

        try:
            for item in edits.get("remove_edges", []) or []:
                if not isinstance(item, dict):
                    continue
                u = item.get("u")
                v = item.get("v")
                if u is None or v is None:
                    continue
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)
                    applied["removed"] += 1
        except Exception as e:
            applied["errors"].append(f"remove_edges: {e}")

        try:
            for item in edits.get("add_edges", []) or []:
                if not isinstance(item, dict):
                    continue
                u = item.get("u")
                v = item.get("v")
                if u is None or v is None:
                    continue
                dist = item.get("distance_km")
                try:
                    dist = float(dist) if dist is not None else None
                except Exception:
                    dist = None
                self.G.add_edge(
                    u,
                    v,
                    weight=dist,
                    link_id=item.get("link_id"),
                    type=item.get("type"),
                )
                applied["added"] += 1
        except Exception as e:
            applied["errors"].append(f"add_edges: {e}")

        # Update positions for any new nodes
        try:
            if self.pos is None or len(self.pos) != self.G.number_of_nodes():
                self.pos = nx.spring_layout(self.G, seed=42)
        except Exception:
            pass

        return applied

    def render_plot(self, plot_spec: dict):
        """Render a matplotlib figure from an LLM "plot" spec.

        Expected keys (all optional):
          - title: str
          - highlight_edges: list[[u,v], ...] OR 'ALL' OR True
          - highlight_nodes: list[node_id,...] OR 'ALL' OR True
          - highlight_color: str

        Behavior:
          - If no highlight_* provided, draw the full topology.
        """
        if not isinstance(plot_spec, dict):
            return {"error": "plot_spec must be a dict"}, None

        title = str(plot_spec.get("title") or "Network Topology")
        highlight_color = str(plot_spec.get("highlight_color") or "red")

        he = plot_spec.get("highlight_edges")
        hn = plot_spec.get("highlight_nodes")

        # Normalize highlight_edges
        highlight_edges = None
        if he is True or (isinstance(he, str) and he.strip().upper() == "ALL"):
            highlight_edges = list(self.G.edges())
        elif isinstance(he, list):
            tmp = []
            for item in he:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    tmp.append((str(item[0]), str(item[1])))
                elif isinstance(item, dict) and item.get("u") is not None and item.get("v") is not None:
                    tmp.append((str(item["u"]), str(item["v"])))
            highlight_edges = tmp if tmp else None

        # Normalize highlight_nodes
        highlight_nodes = None
        if hn is True or (isinstance(hn, str) and hn.strip().upper() == "ALL"):
            highlight_nodes = [str(n) for n in self.G.nodes()]
        elif isinstance(hn, list):
            highlight_nodes = [str(n) for n in hn] if hn else None

        # Always draw the base graph; highlights are optional
        return self._visualize_graph(
            highlight_edges=highlight_edges,
            highlight_nodes=highlight_nodes,
            title=title,
            highlight_color=highlight_color,
        )
