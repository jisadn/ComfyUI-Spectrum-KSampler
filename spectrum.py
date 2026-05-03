"""Spectrum state management, fast-forward path, and shared sampling logic."""

import logging
import math
from typing import Optional, Dict

import torch

import comfy.sample
import comfy.utils
import comfy.patcher_extension
import latent_preview

from .forecaster import SpectrumPredictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DCW: post-step SNR-t bias correction (arXiv:2604.16044). Anima form, λ < 0,
# schedule fixed to one_minus_sigma. See anima_lora/bench/dcw/findings.md.
#
# Applied via two coordinated hooks:
#   1) CALC_COND_BATCH wrapper — mutates x_in in-place at each new-step boundary
#      using the previous step's cached post-CFG denoised. x_in IS the sampler's
#      `x` reference (passed all the way down from KSamplerX0Inpaint), so the
#      Euler/ER-SDE/DPM step that follows operates on the corrected latent.
#   2) sampler_post_cfg_function — captures the post-CFG denoised after each
#      step for use by (1) on the next step.
# ---------------------------------------------------------------------------


class DCWState:
    def __init__(self, lam: float, schedule: str = "one_minus_sigma"):
        self.lam = lam
        self.schedule = schedule
        self.last_denoised: Optional[torch.Tensor] = None
        self.curr_sigma: Optional[float] = None

    def schedule_value(self, sigma_i: Optional[float]) -> float:
        if sigma_i is None:
            return 0.0
        if self.schedule == "one_minus_sigma":
            return 1.0 - sigma_i
        if self.schedule == "sigma_i":
            return sigma_i
        if self.schedule == "const":
            return 1.0
        return 0.0  # "none"


def _make_dcw_calc_cond_batch_wrapper(state: DCWState):
    def wrapper(executor, model, conds, x_in, timestep, model_options):
        # In flow-matching / CONST model_sampling, timestep == sigma.
        sigma = float(timestep[0]) if timestep.ndim else float(timestep)
        new_step = (
            state.curr_sigma is None or abs(sigma - state.curr_sigma) > 1e-8
        )
        if new_step:
            if state.last_denoised is not None and state.lam != 0.0:
                s = state.lam * state.schedule_value(state.curr_sigma)
                if s != 0.0:
                    # In-place: x_in ← x_in + s · (x_in − last_denoised).
                    # x_in IS the sampler's tensor; this propagates to the
                    # Euler/ER-SDE step that runs after the model returns.
                    x_in.add_(x_in - state.last_denoised, alpha=s)
            state.curr_sigma = sigma
        return executor(model, conds, x_in, timestep, model_options)

    return wrapper


def _make_dcw_post_cfg_hook(state: DCWState):
    def hook(args):
        # args["denoised"] is post-CFG x0_pred. Clone so the cache survives
        # downstream in-place ops on the sampler's tensors.
        state.last_denoised = args["denoised"].clone()
        return args["denoised"]

    return hook


def _spectrum_fast_forward(
    dit, timestep: torch.Tensor, predicted_feature: torch.Tensor
) -> torch.Tensor:
    """Runs only t_embedder + final_layer + unpatchify on predicted features.

    Returns the same shape as diffusion_model.forward() — 5D for video DiTs.
    """
    if timestep.ndim == 1:
        timestep = timestep.unsqueeze(1)
    # Replicate the model's two-step t_embedder call: Timesteps (sinusoidal,
    # always float32) -> cast to model dtype -> TimestepEmbedding (linear layers).
    # Calling t_embedder as a single Sequential skips the intermediate cast.
    t_sinusoidal = dit.t_embedder[0](timestep)
    t_emb, adaln = dit.t_embedder[1](t_sinusoidal.to(predicted_feature.dtype))
    t_emb = dit.t_embedding_norm(t_emb)
    # Mod guidance: add cached pooled-text projection from the DIFFUSION_MODEL
    # wrapper.  On actual steps the wrapper computes base+delta from post-adapter
    # context and caches it on dit._mod_pooled_proj.  On cached steps we reuse
    # the last actual step's value (text doesn't change between steps).
    pooled_proj = getattr(dit, "_mod_pooled_proj", None)
    if pooled_proj is not None:
        pp = pooled_proj.unsqueeze(1).to(t_emb.dtype)
        if pp.shape[0] == t_emb.shape[0]:
            t_emb = t_emb + pp
        elif pp.shape[0] == 1:
            t_emb = t_emb + pp.expand_as(t_emb)
    x = dit.final_layer(predicted_feature, t_emb, adaln_lora_B_T_3D=adaln)
    return dit.unpatchify(x)


class SpectrumState:
    def __init__(
        self,
        window_size: float,
        flex_window: float,
        warmup_steps: int,
        w: float,
        m: int,
        lam: float,
        num_steps: int,
    ):
        self.window_size = window_size
        self.flex_window = flex_window
        self.warmup_steps = warmup_steps
        self.w = w
        self.m_param = m
        self.lam = lam
        self.num_steps = num_steps

        # Runtime
        self.step_idx = -1
        self.last_sigma: Optional[float] = None
        self.mode = "actual"
        self.curr_ws = window_size
        self.consec_cached = 0
        self.fwd_count = 0

        # Forecasters keyed by cond_or_uncond value (0=cond, 1=uncond)
        self.forecasters: Dict[int, SpectrumPredictor] = {}
        self.captured_feat: Optional[torch.Tensor] = None

    def should_cache(self) -> bool:
        if self.step_idx < self.warmup_steps:
            return False
        stop_at = self.num_steps - 3
        if self.step_idx >= stop_at:
            return False
        return (self.consec_cached + 1) % max(1, math.floor(self.curr_ws)) != 0

    def has_forecasters(self, cond_or_uncond: list) -> bool:
        return all(cou in self.forecasters for cou in cond_or_uncond)


def _capture_pre_hook(module, args):
    """Module-singleton pre-hook on final_layer — stores the pre-final feature
    on whichever SpectrumState is currently bound to the module.
    """
    state = getattr(module, "_spectrum_state", None)
    if state is not None:
        state.captured_feat = args[0].detach().clone()


def _ensure_capture_hook(dit) -> None:
    final_layer = dit.final_layer
    if getattr(final_layer, "_spectrum_hook_installed", False):
        return
    final_layer.register_forward_pre_hook(_capture_pre_hook)
    final_layer._spectrum_hook_installed = True


def spectrum_sample(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise,
    window_size,
    flex_window,
    warmup_steps,
    blend_w,
    cheby_degree,
    ridge_lambda,
    dcw_lambda: float = -0.010,
    dcw_schedule: str = "one_minus_sigma",
):
    """Shared Spectrum sampling logic used by all node tiers.

    dcw_lambda: DCW bias-correction strength. 0.0 = disabled. Default -0.010
        closes Anima's late-step velocity-norm gap (negative — opposite-sign
        from the paper). See anima_lora/bench/dcw/findings.md.
    """
    m = model.clone()

    # DCW: register CALC_COND_BATCH wrapper + post-CFG hook only when enabled.
    if dcw_lambda != 0.0:
        dcw_state = DCWState(lam=dcw_lambda, schedule=dcw_schedule)
        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
            _make_dcw_calc_cond_batch_wrapper(dcw_state),
            m.model_options,
            is_model_options=True,
        )
        m.set_model_sampler_post_cfg_function(_make_dcw_post_cfg_hook(dcw_state))

    state = SpectrumState(
        window_size=window_size,
        flex_window=flex_window,
        warmup_steps=warmup_steps,
        w=blend_w,
        m=cheby_degree,
        lam=ridge_lambda,
        num_steps=steps,
    )

    dit = m.model.diffusion_model
    model_sampling = m.model.model_sampling

    # Install capture hook once per DiT instance (no-op on subsequent runs) and
    # bind this sample's state to the module. The hook reads state from the
    # module attribute, so its identity/closure is stable across samples —
    # torch.compile's dynamo cache survives between runs.
    _ensure_capture_hook(dit)
    dit.final_layer._spectrum_state = state

    old_wrapper = m.model_options.get("model_function_wrapper")

    def spectrum_wrapper(apply_model, args):
        input_x = args["input"]
        timestep = args["timestep"]
        c = args["c"]
        cond_or_uncond = args["cond_or_uncond"]

        sigma_val = timestep[0].item()

        if state.last_sigma is None or abs(sigma_val - state.last_sigma) > 1e-8:
            if state.step_idx >= 0:
                if state.mode == "actual":
                    state.fwd_count += 1
                    if state.step_idx >= state.warmup_steps:
                        state.curr_ws = round(state.curr_ws + state.flex_window, 3)
                    state.consec_cached = 0
                else:
                    state.consec_cached += 1

            state.step_idx += 1
            state.last_sigma = sigma_val
            state.mode = "cached" if state.should_cache() else "actual"

        if state.mode == "cached" and state.has_forecasters(cond_or_uncond):
            predictions = []
            for cou in cond_or_uncond:
                pred_feat = state.forecasters[cou].predict(float(state.step_idx))
                predictions.append(pred_feat)

            batched_feat = torch.cat(predictions, dim=0)
            t_internal = model_sampling.timestep(timestep).to(batched_feat.dtype)
            noise_pred = _spectrum_fast_forward(dit, t_internal, batched_feat)
            return model_sampling.calculate_denoised(
                timestep, noise_pred.float(), input_x
            )

        state.mode = "actual"

        if old_wrapper is not None:
            result = old_wrapper(apply_model, args)
        else:
            result = apply_model(input_x, timestep, **c)

        feat = state.captured_feat
        if feat is not None:
            batch_chunks = len(cond_or_uncond)
            feat_chunks = feat.chunk(batch_chunks, dim=0)
            for idx, cou in enumerate(cond_or_uncond):
                if cou not in state.forecasters:
                    state.forecasters[cou] = SpectrumPredictor(
                        state.m_param,
                        state.lam,
                        state.w,
                        feat.device,
                        feat_chunks[idx].shape,
                        state.num_steps,
                    )
                state.forecasters[cou].update(float(state.step_idx), feat_chunks[idx])

        return result

    m.set_model_unet_function_wrapper(spectrum_wrapper)

    latent_img = latent_image["samples"].clone()
    latent_img = comfy.sample.fix_empty_latent_channels(
        m, latent_img, latent_image.get("downscale_ratio_spacial")
    )

    batch_inds = latent_image.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_img, seed, batch_inds)

    noise_mask = latent_image.get("noise_mask")
    callback = latent_preview.prepare_callback(m, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    try:
        samples = comfy.sample.sample(
            m,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_img,
            denoise=denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
    finally:
        dit.final_layer._spectrum_state = None
        if hasattr(dit, "_mod_pooled_proj"):
            del dit._mod_pooled_proj

    if state.step_idx >= 0:
        if state.mode == "actual":
            state.fwd_count += 1
        else:
            state.consec_cached += 1

    actual = state.fwd_count
    total = state.step_idx + 1
    speedup = total / max(1, actual)
    do_cfg = not math.isclose(cfg, 1.0)
    cfg_note = " (x2 for CFG)" if do_cfg else ""
    logger.info(
        f"Spectrum: {actual}/{total} actual forwards "
        f"({speedup:.2f}x theoretical speedup{cfg_note})"
    )

    out = latent_image.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return (out,)
