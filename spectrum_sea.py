"""Spectral-Evolution-Aware (SEA) cache-decision metric + auto-δ calibration.

ComfyUI-side port of ``networks/spectrum_sea.py`` from the Anima training repo.
The Chebyshev feature *forecasting* (the reuse path) in ``spectrum.py`` is
unchanged — SEA only swaps the *when-to-skip* trigger: instead of the
content-blind growing window, refresh when the accumulated SEA-filtered latent
distance since the last refresh crosses a threshold δ. δ is auto-calibrated to a
target refresh fraction on the first generate per config and cached to disk, so
later generates at the same config get the fast SEA path with no extra cost.

SEA filter (SeaCache, Chung et al., arXiv:2602.18993v2, §4.1) — a σ-dependent
Wiener-like gain that keeps low-frequency *content* and attenuates high-frequency
*noise*, so a cache distance measured on filtered latents tracks content
evolution rather than stochastic detail:

    G_t(f)      = a·S_x(f) / (a²·S_x(f) + b²),   a = 1−σ, b = σ,  S_x(f) ∝ f^{-β}
    G_t^norm(f) = G_t(f) / mean_f G_t(f)
    P_t(I)      = iFFT( G_t^norm(f) ⊙ FFT(I) )

The δ generalizes across SMC-CFG / mod-guidance (validated in the training repo's
bench/spectrum_sea/guidance_generalization.py — <0.5% trace perturbation, δ ratio
1.00×), which is why the calibration pass can run with mod-guidance active (this
node always does) and the cached δ stays valid.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# SEA filter (verbatim port of networks/spectrum_sea.py)
# --------------------------------------------------------------------------- #
def radial_freq(h: int, w: int, device, dtype) -> torch.Tensor:
    """Radial spatial-frequency magnitude grid for ``fft2`` over ``(h, w)``."""
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(gy * gy + gx * gx)


def sea_gain(h: int, w: int, sigma: float, beta: float, device, dtype) -> torch.Tensor:
    """Density-normalized SEA gain ``G_t^norm(f)`` as an ``(h, w)`` mask."""
    a = 1.0 - float(sigma)
    b = float(sigma)
    f = radial_freq(h, w, device, dtype)
    f_floor = 1.0 / max(h, w)
    sx = f.clamp_min(f_floor) ** (-beta)
    g = a * sx / (a * a * sx + b * b + _EPS)
    g = g / g.mean().clamp_min(_EPS)
    return g


def sea_filter(x: torch.Tensor, sigma: float, beta: float = 2.0) -> torch.Tensor:
    """Apply the SEA filter ``P_t`` over the spatial (last two) axes of ``x``."""
    assert x.dim() >= 2, f"expected (..., H, W), got {tuple(x.shape)}"
    h, w = x.shape[-2], x.shape[-1]
    g = sea_gain(h, w, sigma, beta, x.device, torch.float32)
    xf = torch.fft.fft2(x.to(torch.float32), dim=(-2, -1))
    yf = xf * g
    y = torch.fft.ifft2(yf, dim=(-2, -1)).real
    return y.to(x.dtype)


def l1rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L1 distance ``‖a − b‖₁ / (‖b‖₁ + ξ)`` — SeaCache Eq. 3."""
    return float((a - b).abs().sum() / (b.abs().sum() + _EPS))


# --------------------------------------------------------------------------- #
# Auto-δ calibration
# --------------------------------------------------------------------------- #
def count_refreshes(dists: Sequence[float], delta: float) -> int:
    """Refreshes the accumulate-until-δ rule fires over ``dists`` (non-increasing in δ)."""
    accum = 0.0
    n = 0
    for d in dists:
        accum += float(d)
        if accum >= delta:
            n += 1
            accum = 0.0
    return n


def solve_delta_for_refresh_ratio(
    dists: Sequence[float], refresh_ratio: float, iters: int = 60
) -> float:
    """δ such that ``count_refreshes(dists, δ)`` ≈ ``round(refresh_ratio·len)``."""
    dists = [float(d) for d in dists]
    n = len(dists)
    if n == 0:
        return _EPS
    target = max(1, round(float(refresh_ratio) * n))
    lo, hi = 0.0, max(sum(dists), _EPS)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if count_refreshes(dists, mid) > target:
            lo = mid
        else:
            hi = mid
    return max(hi, _EPS)


def window_decision_fraction(
    num_steps: int,
    warmup_steps: int,
    stop_at: int,
    window_size: float,
    flex_window: float,
) -> float:
    """Refresh fraction the growing-window schedule spends in the decision region.

    The SEA auto-δ target defaults to this, so SEA is a like-for-like swap at
    matched compute for any step count (do NOT hard-code 0.62 — that's only the
    24-step value and over-computes elsewhere).
    """
    curr_ws = window_size
    consec = 0
    actual_dec = 0
    n_dec = 0
    for i in range(num_steps):
        if i < warmup_steps or i >= stop_at:
            actual = True
        else:
            actual = (consec + 1) % max(1, math.floor(curr_ws)) == 0
            n_dec += 1
            actual_dec += int(actual)
        if actual:
            if i >= warmup_steps:
                curr_ws = round(curr_ws + flex_window, 3)
            consec = 0
        else:
            consec += 1
    return actual_dec / max(1, n_dec)


# --------------------------------------------------------------------------- #
# δ disk cache (ComfyUI user/output dir)
# --------------------------------------------------------------------------- #
def _cache_path() -> str:
    """Resolve a persistent path for the δ cache JSON.

    Prefers ComfyUI's user dir (survives node reinstalls, user-discoverable),
    falls back to the output dir, then the node folder for non-ComfyUI contexts.
    """
    try:
        import folder_paths  # type: ignore

        get_user = getattr(folder_paths, "get_user_directory", None)
        base = get_user() if callable(get_user) else folder_paths.get_output_directory()
    except Exception:
        base = os.path.dirname(__file__)
    return os.path.join(base, "spectrum_sea_delta.json")


def make_cache_key(
    num_steps: int,
    warmup_steps: int,
    stop_at: int,
    refresh_ratio: float,
    cfg: float,
    sampler: str,
    h: int,
    w: int,
    extra: str = "",
) -> str:
    """Stable string key. Mirrors the training repo's tuple (prompt deliberately
    excluded — fixed δ + per-prompt-varying refresh pattern is the design).

    ``extra`` carries any non-default substrate that changes the trajectory the
    δ is calibrated against (CFG++ λ, FSG band/K) so its δ never aliases a plain
    run's. Empty (the common case) reproduces the original 8-field key exactly.
    """
    fields = [
        int(num_steps),
        int(warmup_steps),
        int(stop_at),
        round(float(refresh_ratio), 4),
        round(float(cfg), 3),
        str(sampler),
        int(h),
        int(w),
    ]
    if extra:
        fields.append(str(extra))
    return "|".join(str(x) for x in fields)


def load_delta(key: str) -> Optional[float]:
    path = _cache_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        v = data.get(key)
        return float(v) if v is not None else None
    except Exception as e:  # corrupt/locked cache must never break a generate
        logger.warning("Spectrum SEA: δ cache read failed (%s); recalibrating.", e)
        return None


def save_delta(key: str, value: float) -> None:
    path = _cache_path()
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:
            data = {}
    data[key] = float(value)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("Spectrum SEA: δ cache write failed (%s); not persisted.", e)


def accumulate_distances(seas: Sequence[torch.Tensor]) -> List[float]:
    """Per-step consecutive SEA distances (offline convenience / tests)."""
    return [l1rel(seas[i], seas[i - 1]) for i in range(1, len(seas))]
