"""qot_utils.py

Minimal QoT utilities used by the demo app.

The original project referenced a richer QoT model. For the benchmark questions in this project,
we only need propagation delay (and a small fiber reference table).

Propagation delay model:
    delay = distance / v_g
    v_g ≈ c / n_eff

Defaults:
- SSMF: n_eff = 1.468  (typical silica)
- HCF:  n_eff = 1.10   (illustrative hollow-core fiber; faster than silica)

All outputs are JSON-friendly floats.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

C_M_PER_S = 299_792_458.0

FIBER_REFERENCE: Dict[str, Dict[str, float]] = {
    "SSMF": {"n_eff": 1.468},
    "HCF": {"n_eff": 1.10},
}

def calc_qot_metrics(distance_km: float, fiber_type: str = "SSMF", optical_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a small metrics dict with propagation delay.

    Args:
        distance_km: Path distance in km.
        fiber_type: "SSMF" or "HCF" (case-insensitive). Unknown types fall back to SSMF.
        optical_params: accepted for API compatibility; unused in this minimal version.
    """
    try:
        d_km = float(distance_km)
    except Exception:
        d_km = 0.0

    ft = str(fiber_type or "SSMF").upper()
    n_eff = float(FIBER_REFERENCE.get(ft, FIBER_REFERENCE["SSMF"])["n_eff"])
    v = C_M_PER_S / n_eff  # m/s
    delay_s = (d_km * 1000.0) / v if v > 0 else 0.0
    delay_ms = delay_s * 1000.0

    return {
        "Fiber Type": ft,
        "Distance (km)": d_km,
        "Total Propagation Delay (ms)": float(delay_ms),
    }
