"""
QoT Utility Functions — Physics-Based Optical Calculations
"""

import math

# ---------------------- GLOBAL OPTICAL CONSTANTS ----------------------
OPTICAL_CONSTANTS = {
    # Fiber / components
    "fiber_att_dB_per_km": 0.2,           # fiber attenuation (dB/km)
    "roadm_loss_dB": 10,                  # ROADM insertion loss per node (dB)
    "amp_noise_figure_dB": 6,             # Amplifier / ILA NF (dB)
    "tx_power_dBm": 0,                    # Per-channel Tx power (dBm)

    # Channel plan / bandwidth
    "symbol_rate_GBd": 64,                # Rs (GBd)
    "roll_off": 0.1,                      # beta
    "Bn_GHz": 64 * (1 + 0.1),             # in-band bandwidth GHz = Rs*(1+β) = 70.4 GHz

    # Reference OSNR bandwidth
    "osnr_reference_bw_Hz": 12.5e9,       # 12.5 GHz (≈0.1 nm)

    # Physical constants
    "wavelength_nm": 1550,
    "optical_freq_Hz": 1.93414e14,        # c / λ
    "planck_Js": 6.62607e-34,
}

# ---------------------- Basic helpers ----------------------
def dBm_to_mW(p_dBm: float) -> float:
    return 10 ** (p_dBm / 10.0)

def mW_to_dBm(p_mW: float) -> float:
    return 10.0 * math.log10(max(p_mW, 1e-20))

# ---------------------- ASE noise (per amplifier) ----------------------
def calc_ase_noise(gain_dB: float, nf_dB: float, B_ref_Hz: float | None = None) -> float:
    """
    ASE noise power (Watts) for a single optical amplifier over reference bandwidth.

    P_ASE = n_sp * h * nu * (G - 1) * B_ref

    - gain_dB : amplifier gain in dB
    - nf_dB   : amplifier noise figure in dB
    - B_ref_Hz: OSNR reference bandwidth (default 12.5 GHz)
    """
    if B_ref_Hz is None:
        B_ref_Hz = OPTICAL_CONSTANTS["osnr_reference_bw_Hz"]

    gain_lin = 10 ** (gain_dB / 10.0)
    nf_lin = 10 ** (nf_dB / 10.0)

    # spontaneous emission factor
    n_sp = nf_lin / 2.0

    h = OPTICAL_CONSTANTS["planck_Js"]
    nu = OPTICAL_CONSTANTS["optical_freq_Hz"]

    P_ase_W = n_sp * h * nu * (gain_lin - 1.0) * B_ref_Hz  # Watts
    return P_ase_W

# ---------------------- Simple path-level metrics ----------------------
def calc_qot_metrics(
    distance_km: float,
    att_per_km: float = OPTICAL_CONSTANTS["fiber_att_dB_per_km"],
    delay_per_km_ms: float = 0.005,
    optical_params: dict | None = None
) -> dict:
    """
    Lightweight attenuation + delay model per edge or path segment.
    (Detailed OSNR/GSNR/Q/BER is computed in QoTAnalyzer.)
    """
    total_atten_dB = float(distance_km) * float(att_per_km)
    total_delay_ms = float(distance_km) * float(delay_per_km_ms)

    gain_dB = 0.0
    nf_dB = 0.0

    if optical_params:
        # Insertion loss of ROADMs (already summed if provided)
        roadm_loss = float(optical_params.get("roadm_insertion_loss_dB", 0.0))
        total_atten_dB += roadm_loss

        # Aggregate link gain / NF if present
        gain_dB += float(optical_params.get("gain_dB", 0.0))
        nf_dB += float(optical_params.get("noise_figure_dB", 0.0))

    net_atten_dB = total_atten_dB - gain_dB

    return {
        "Total Attenuation (dB)": round(net_atten_dB, 2),
        "Total Propagation Delay (ms)": round(total_delay_ms, 4),
        "Gain (dB)": round(gain_dB, 2),
        "Noise Figure (dB)": round(nf_dB, 2),
    }
