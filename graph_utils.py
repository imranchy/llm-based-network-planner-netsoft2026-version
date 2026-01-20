# ----------------- Imports -----------------
import pandas as pd
import os
from functools import wraps
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st


# ----------------- Utility Functions -----------------
def ensure_results_dir(dir_name="results"):
    """
    Ensure the results directory exists and return its path.
    """
    results_dir = os.path.abspath(dir_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# ----------------- Interactive Plot Decorator -----------------
def interactive_plot(func):
    """
    Decorator for visualizing plots.
    Ensures consistent return format (info, fig) for all visualization functions.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Normalize return structure
        if isinstance(result, tuple):
            # Extract first dict-like and first matplotlib Figure
            import matplotlib.figure as mfig
            info, fig = None, None
            for r in result:
                if isinstance(r, dict):
                    info = r
                elif isinstance(r, mfig.Figure):
                    fig = r
            info = info or {}
            return info, fig
        else:
            return {}, result
    return wrapper


# ----------------- Node Resolver -----------------
def resolve_nodes(G, src, dst):
    """
    Safely resolve source and destination nodes, ensuring both exist in the graph.
    """
    src = str(src).strip() if src is not None else None
    dst = str(dst).strip() if dst is not None else None

    if src is None or dst is None:
        print("❌ Source or destination cannot be empty.")
        return None, None

    if src not in G.nodes:
        print(f"❌ Source node '{src}' not found. Available nodes: {list(G.nodes)[:10]}...")
        return None, None

    if dst not in G.nodes:
        print(f"❌ Destination node '{dst}' not found. Available nodes: {list(G.nodes)[:10]}...")
        return None, None

    return src, dst


# ----------------- Graph Visualization -----------------
@interactive_plot
def _visualize_graph_custom(G, pos=None, highlight_edges=None, title="Network Graph",
                            highlight_color="red"):
    """
    Visualize a NetworkX graph with optional highlighted edges and equipment info.
    Fully Streamlit-ready (returns info + matplotlib Figure only).

    Parameters:
        G : networkx.Graph
        pos : dict (optional)
        highlight_edges : list of tuples
        title : str
        highlight_color : str

    Returns:
        (info, fig)
    """
    highlight_edges = highlight_edges or []
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw(G, pos, with_labels=True, node_color="skyblue",
            node_size=900, font_size=9, ax=ax, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.2, ax=ax)

    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges,
                               edge_color=highlight_color, width=2.8, ax=ax)

    # Edge labels with distance & equipment info
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        link_id = d.get("link_id", "")
        weight = d.get("weight", "?")
        equipment = d.get("equipment", "")
        label = f"{link_id} ({weight} km)"
        if equipment:
            label += f" | {equipment}"
        edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title(title)
    ax.axis("off")

    # Summary info (used by all classes)
    info = {
        "title": title,
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "highlighted_edges_count": len(highlight_edges),
    }

    return info, fig
