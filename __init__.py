"""Spectrum: Adaptive Spectral Feature Forecasting for ComfyUI.

Drop-in KSampler replacement that accelerates diffusion sampling via
Chebyshev polynomial feature forecasting (Han et al., CVPR 2026).

On "actual" steps the full model runs and block outputs are captured.
On "cached" steps all transformer blocks are skipped — only t_embedder +
final_layer + unpatchify execute, using predicted features from a
Chebyshev ridge-regression fit. Works with any ComfyUI sampler (Euler,
DPM, er_sde, etc.) because caching is handled transparently inside the
model_function_wrapper.

Node tiers:
  - SpectrumKSampler: unified drop-in — acceleration + SEA scheduling
    (refresh_ratio dial; -1 = off) + modulation guidance (mod_w_profile;
    'off' = none). Subsumes the former basic / ModGuidance / SEA samplers,
    whose class keys remain as hidden deprecated aliases.
  - SpectrumKSamplerAdvanced: + full Spectrum tuning, raw mod-guidance scalars,
    adapter selection, and DCW
  - SpectrumSPDKSampler (SPEED): + SPD multi-resolution prefix (low-res early,
    spectral-expand to full res at the handoff, Spectrum-forecasted tail)
  - SpectrumSPDLoRAKSampler: SPEED with the resolution schedule auto-read from
    an SPD-trained LoRA's metadata
  - AnimaModGuidance: standalone mod-guidance model patcher (composes with any sampler)
  - DiTSpectrumPatch: standalone Spectrum MODEL patcher (no sampling)

Also registers an `er_sde_cns` entry in the global sampler dropdown: Colored
Noise Sampling (Davidson et al., arXiv 2605.30332) — the ER-SDE solver with
per-step noise recolored toward unresolved frequency bands. Pick it in any
KSampler's `sampler_name` field; composes with the Spectrum / mod-guidance
patches above. Anima-calibrated γ; auto-downloaded on first use.
"""

from .cns import register_cns_sampler
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

register_cns_sampler()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
