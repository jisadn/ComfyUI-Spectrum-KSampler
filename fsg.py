"""Foresight Guidance (FSG) + CFG++ substrate — ComfyUI port.

FSG reframes CFG as a *fixed-point calibration*: at scheduled mid-σ steps it
runs ``K`` forward(guided)–backward(unconditional) iterations over a long
interval ``Δσ`` to pull ``x_t → x̂_t`` onto the golden path where the conditional
and unconditional velocities agree, then the sampler step proceeds from ``x̂_t``.
Training-free, checkpoint-agnostic, deterministic.

Paper: "Towards a Golden Classifier-Free Guidance Path via Foresight Fixed
Point Iterations" (NeurIPS 2025, arXiv 23177). The paper is ε-prediction + DDIM;
Anima is velocity-prediction flow-matching, so the forward-backward operator maps
onto the reversible Euler ODE (no DDIM machinery):

    v^γ  = v^u + γ·(v^c − v^u)              # CFG-guided velocity
    x'   = x  − Δσ · v^γ(x,   σ)            # denoise σ → σ−Δσ (guided)
    x''  = x' + Δσ · v^u(x',  σ−Δσ)         # re-noise back (unconditional)
    F(x) = x'' ;  iterate x ← F(x), K times

This is a hand-mirror of ``anima_lora/library/inference/corrections/fsg.py``
(and the CFG++ reweight from ``library/inference/sampling.py``). The operator
math is identical; only the velocity source differs — here velocities come from
ComfyUI's ``comfy.samplers.calc_cond_batch`` (cond/uncond denoised → v-space)
rather than the library's direct ``anima(x, t, embed)`` call.

**Production config (the validated point, 1024 tier / 28-step er_sde):**
substrate CFG++ λ=1.5, band [0.59, 0.75], K=3, Δσ=0.1, γ=guidance(=4). The
contracting band moves DOWN for more steps and for low-token (~768px) renders,
UP for fewer steps — re-tune if you change steps/resolution (σ≈0.94 always
diverges; the paper's noisy-stage prescription is wrong on Anima).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

import comfy.patcher_extension
import comfy.samplers

logger = logging.getLogger(__name__)


# --- CFG++ substrate ------------------------------------------------------


def cfgpp_guidance_weight(sigma_i: float, sigma_next: float, lam: float) -> float:
    """CFG++ effective guidance weight for one step (integrator-agnostic).

    CFG++ differs from CFG *only* in guidance-strength scheduling (paper App
    A.2): both add ``weight·(v^c − v^u)`` to the update, CFG with a constant
    ``w``, CFG++ with the σ-scheduled weight. The flow-matching form is

        w_eff = λ · (1 − σ_next) · σ_i / (σ_i − σ_next)

    Applied as ``noise_pred = v^u + w_eff·(v^c − v^u)`` it composes with any
    integrator (Euler, ER-SDE): the sampler consumes the reweighted prediction
    unchanged. At the final step (σ_next → 0) it collapses to ``w_eff = λ``.
    """
    ds = sigma_i - sigma_next
    if ds <= 0.0:
        return lam
    return lam * (1.0 - sigma_next) * sigma_i / ds


class CFGPPState:
    """σ-scheduled CFG++ reweight installed as a ``sampler_cfg_function``.

    Holds the sampler's σ schedule so a given step's σ_i can be mapped to its
    σ_next (the weight needs both). The combine runs in denoised space — which
    is affine in the noise prediction, so reweighting the denoised cond/uncond
    is identical to reweighting the velocities.
    """

    def __init__(self, lam: float, sigmas: List[float]):
        self.lam = float(lam)
        # Descending σ schedule incl. the trailing 0.0 (len == steps + 1).
        self.sigmas = [float(s) for s in sigmas]

    def _sigma_next(self, sig: float) -> float:
        sched = self.sigmas
        if len(sched) < 2:
            return 0.0
        idx = min(range(len(sched) - 1), key=lambda i: abs(sched[i] - sig))
        return sched[idx + 1]


def _make_cfgpp_cfg_function(state: CFGPPState):
    """Build a ComfyUI ``sampler_cfg_function`` applying the CFG++ reweight.

    Contract (``comfy.samplers.cfg_function``): ``cfg_result = x − fn(args)``,
    so we return ``x − denoised_combined``.
    """

    def cfg_function(args):
        cond_denoised = args["cond_denoised"]
        uncond_denoised = args["uncond_denoised"]
        x_in = args["input"]
        sigma = args.get("sigma", args.get("timestep"))
        sig = float(sigma.flatten()[0]) if torch.is_tensor(sigma) else float(sigma)

        w_eff = cfgpp_guidance_weight(sig, state._sigma_next(sig), state.lam)
        denoised_combined = uncond_denoised + w_eff * (cond_denoised - uncond_denoised)
        return x_in - denoised_combined

    return cfg_function


def install_cfgpp(model_patcher, *, lam: float, sigmas: List[float]) -> None:
    """Replace the CFG combine with the CFG++ σ-scheduled reweight.

    Mutually exclusive with SMC-CFG (both own ``sampler_cfg_function``); the
    caller must refuse to install both. Caller must clone the patcher first.
    """
    state = CFGPPState(lam=lam, sigmas=sigmas)
    model_patcher.set_model_sampler_cfg_function(_make_cfgpp_cfg_function(state))


# --- FSG fixed-point calibrator -------------------------------------------


def _clean_model_options(model_options: dict) -> dict:
    """A model_options that bypasses the Spectrum cache + the FSG/DCW wrappers.

    FSG's own velocity forwards must hit the *real* DiT (never a Spectrum
    forecast) and must not re-enter FSG (recursion) or DCW (it would mis-detect
    a new step at σ−Δσ and corrupt its state). We strip the
    ``model_function_wrapper`` (Spectrum) and the CALC_COND_BATCH wrapper list
    (FSG/DCW) while KEEPING the DIFFUSION_MODEL wrappers — so mod guidance still
    applies, matching the library's FSG forwards (which run on the live model).
    """
    opts = dict(model_options)
    opts.pop("model_function_wrapper", None)
    to = model_options.get("transformer_options")
    if to is not None:
        to_copy = dict(to)
        wrappers = to_copy.get("wrappers")
        if wrappers is not None:
            w_copy = dict(wrappers)
            w_copy.pop(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, None)
            to_copy["wrappers"] = w_copy
        opts["transformer_options"] = to_copy
    return opts


@dataclass
class FSGCalibrator:
    """Forward-backward fixed-point calibrator, scheduled over a σ-band.

    Args mirror the library plugin: ``band`` (σ_lo, σ_hi) gate, ``k`` iterations
    (k=0 inert), ``d_sigma`` the forward-backward stride Δσ (too large → σ≈0.94
    diverges), ``gamma`` the calibration guidance (None → outer guidance_scale).
    """

    band: Tuple[float, float] = (0.59, 0.75)
    k: int = 3
    d_sigma: float = 0.1
    gamma: Optional[float] = None

    def __post_init__(self) -> None:
        self.k = int(self.k)
        self.d_sigma = float(self.d_sigma)
        self.band = (float(self.band[0]), float(self.band[1]))

    def scheduled(self, sigma_i: float) -> bool:
        lo, hi = self.band
        return self.k > 0 and lo <= float(sigma_i) <= hi

    @torch.no_grad()
    def calibrate(
        self,
        model,
        conds,
        x: torch.Tensor,
        sigma_i: float,
        clean_options: dict,
        guidance_scale: float,
        timestep_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``x̂`` after K forward-backward iterations.

        ``conds == [cond, uncond]`` (the list ComfyUI's ``sampling_function``
        hands to ``calc_cond_batch``). Velocities are recovered from the
        denoised predictions: for CONST/flow model_sampling ``v = (x − x0)/σ``.
        Costs ~3·K extra forwards per scheduled step (cond+uncond at σ batched,
        uncond at σ−Δσ).
        """
        gamma = float(guidance_scale) if self.gamma is None else float(self.gamma)
        ds = self.d_sigma
        s_lo = max(float(sigma_i) - ds, 1e-3)
        uncond = conds[1]

        cur = x.clone()
        for _ in range(self.k):
            t_i = timestep_ref.new_full((cur.shape[0],), float(sigma_i))
            out = comfy.samplers.calc_cond_batch(model, conds, cur, t_i, clean_options)
            vc = (cur - out[0]) / sigma_i
            vu = (cur - out[1]) / sigma_i
            vg = vu + gamma * (vc - vu)
            x_fwd = cur - ds * vg  # denoise σ → σ−Δσ (guided)

            t_lo = timestep_ref.new_full((cur.shape[0],), s_lo)
            out_lo = comfy.samplers.calc_cond_batch(
                model, [uncond], x_fwd, t_lo, clean_options
            )
            vu_lo = (x_fwd - out_lo[0]) / s_lo
            cur = x_fwd + ds * vu_lo  # invert back (uncond)
        return cur.to(x.dtype)


def fsg_step_indices(fsg: FSGCalibrator, sigmas: List[float], num_steps: int) -> frozenset:
    """The step indices whose σ lands in the FSG band — forced actual by Spectrum."""
    return frozenset(
        i for i in range(min(num_steps, len(sigmas))) if fsg.scheduled(float(sigmas[i]))
    )


class _FSGWrapperState:
    def __init__(self, fsg: FSGCalibrator, guidance_scale: float):
        self.fsg = fsg
        self.guidance_scale = guidance_scale
        self.last_sigma: Optional[float] = None


def _make_fsg_calc_cond_batch_wrapper(state: _FSGWrapperState):
    def wrapper(executor, model, conds, x_in, timestep, model_options):
        sigma = float(timestep.flatten()[0]) if timestep.ndim else float(timestep)
        new_step = state.last_sigma is None or abs(sigma - state.last_sigma) > 1e-8
        if new_step:
            state.last_sigma = sigma
            # FSG needs the cond/uncond gap, so it only fires when an uncond
            # branch is present (CFG ≠ 1) and the step is in-band.
            if (
                state.fsg.scheduled(sigma)
                and len(conds) >= 2
                and conds[1] is not None
            ):
                clean = _clean_model_options(model_options)
                x_hat = state.fsg.calibrate(
                    model, conds, x_in, sigma, clean, state.guidance_scale, timestep
                )
                # In-place: x_in IS the sampler's x reference (calc_cond_batch
                # tiles it across the cond batch), so the step proceeds from x̂.
                x_in.copy_(x_hat.to(x_in.dtype))
        return executor(model, conds, x_in, timestep, model_options)

    return wrapper


def install_fsg(model_patcher, *, fsg: FSGCalibrator, guidance_scale: float) -> None:
    """Register the FSG pre-step calibration wrapper on a cloned ModelPatcher.

    No-op when ``fsg`` is None or inert (k=0). Caller must clone the patcher and
    must also force the FSG-scheduled steps to actual Spectrum forwards (the
    calibrated latent is meaningless against a cached feature forecast).
    """
    if fsg is None or fsg.k <= 0:
        return
    state = _FSGWrapperState(fsg, guidance_scale)
    comfy.patcher_extension.add_wrapper(
        comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
        _make_fsg_calc_cond_batch_wrapper(state),
        model_patcher.model_options,
        is_model_options=True,
    )
