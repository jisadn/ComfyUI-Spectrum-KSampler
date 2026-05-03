# Spectrum for ComfyUI

Training-free diffusion sampling acceleration via **Chebyshev polynomial feature forecasting** ([Han et al., CVPR 2026](https://arxiv.org/abs/2603.01623)). Drop-in KSampler replacement that skips transformer blocks on predicted steps for ~2-3x speedup.

## How it works

Standard diffusion runs the full DiT (all transformer blocks) at every denoising step. Spectrum observes that block outputs are smooth functions of the timestep, so most steps can be **predicted** instead of computed.

On "actual" steps the full model runs and block outputs are captured. On "cached" steps all transformer blocks are skipped — only `t_embedder` + `final_layer` + `unpatchify` execute, using features predicted from a Chebyshev ridge-regression fit.

### Adaptive window schedule

The window size N starts at `window_size` and grows by `flex_window` after each actual forward:

1. **Warmup** (first N steps): always run full forward to seed the forecaster
2. **Adaptive**: actual forward every `floor(N)` cached steps; N grows after each forward

With 28 steps and defaults: ~**8 actual forwards** out of 28 total steps.

## Usage

Place the **KSampler (Spectrum)** node where you'd normally use a KSampler. It has the same inputs (model, seed, steps, cfg, sampler, scheduler, conditioning, latent) plus Spectrum-specific parameters.

Works with any ComfyUI sampler (Euler, DPM, er_sde, etc.) because caching is handled transparently inside a model function wrapper. Chains with other model wrappers (Flex Attention, Flash Attention 4, etc.).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 2.0 | Initial caching window N |
| `flex_window` | 0.25 | Window growth rate per actual forward |
| `warmup_steps` | 6 | Steps that always run full forward |
| `blend_w` | 0.3 | Chebyshev/Taylor blend weight (1.0 = pure Chebyshev) |
| `cheby_degree` | 3 | Number of Chebyshev basis functions |
| `ridge_lambda` | 0.1 | Ridge regression regularization strength |

### Tuning tips

- **More speedup**: increase `flex_window` (faster window growth = fewer forwards)
- **Better quality**: increase `warmup_steps`, decrease `flex_window`
- **Aggressive acceleration**: `flex_window=1.0`, `blend_w=0.7` (~3-4x speedup)

## Modulation guidance

The **KSampler (Spectrum + Mod Guidance)** and **Advanced** variants add text-conditioned quality steering via a learned `pooled_text_proj` MLP adapter ([Starodubcev et al., ICLR 2026](https://arxiv.org/abs/2502.15349)). The adapter projects pooled text embeddings into a guidance delta that is injected into the DiT's AdaLN timestep embedding, steering generation toward the specified quality attributes.

The default ~12MB `pooled_text_proj` weight is auto-downloaded on first use from the [anima_lora release page](https://github.com/sorryhyun/anima_lora/releases/tag/mod_guidance) into `ComfyUI/models/anima_mod_guidance/`. The simple node always uses the default; the advanced node exposes an adapter dropdown where `(auto-download default)` triggers the same download or you can pick a custom adapter from `loras/`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip` | — | CLIP encoder for encoding quality tags |
| `adapter` | `(auto-download default)` | `pooled_text_proj` safetensors file (advanced node only) |
| `quality_tags` | `absurdres, highres, masterpiece, ...` | Quality/aesthetic tags to steer toward |
| `mod_w_profile` (simple) | `step_i8_skip27` | Per-block guidance preset. `step_i8_skip27` (default, best quality) protects blocks 0–7 + 27 and applies `w=3` to blocks 8–26. `step_i14` is the safe option — use it when a LoRA shows anatomy drift. `uniform_w3` recovers pre-0413 legacy behavior. |
| `mod_w` (advanced) | 3.0 | Peak guidance strength applied per-block |
| `mod_start_layer` (advanced) | 8 | First block (inclusive) that receives the steering delta. `0` = uniform legacy behavior |
| `mod_end_layer` (advanced) | -1 | Last block + 1 (exclusive). `-1` = all remaining blocks. Set to `27` to skip Anima's compensation block |
| `mod_taper` (advanced) | 0 | Number of late slots to scale by `mod_taper_scale`. `0` disables taper |
| `mod_taper_scale` (advanced) | 0.25 | Multiplier for tapered slots |
| `mod_final_w` (advanced) | 0.0 | `w` applied at `final_layer`. `0` = don't disturb the output head |

Per-block guidance schedules address quality drift on LoRAs whose distribution sits far from the positive-prompt axis (e.g. early blocks blowing out tonal DC into uniform color collapse). The default `step_i8_skip27` protects blocks 0–7 and the final compensation block 27 from the steering delta while keeping the base text projection uniform across all blocks. See `docs/mod-guidance.md` in the anima_lora repo for the underlying rationale.

## DCW post-step bias correction

All three nodes expose `dcw_lambda` and `dcw_band_mask` widgets that toggle **DCW** ([Yu et al., CVPR 2026](https://arxiv.org/abs/2604.16044)), a sampler-level post-step correction for the SNR-t bias of flow-matching DiTs. Each step's `prev_sample` is mixed toward (or away from) the post-CFG `x0_pred`, optionally restricted to a single-level Haar subband of the differential:

```
diff           = x_{i+1} − x0_pred_i
diff_masked    = haar_idwt(mask(haar_dwt(diff)))   # band restriction
x_{i+1}       += λ · (1 − σ_i) · diff_masked
```

| `dcw_lambda` | Behavior |
|---|---|
| `-0.015` (default) | Tuned for `dcw_band_mask = LL` — closes ≈42% of Anima's late-half integrated \|gap\|. **Negative** — opposite-sign from the paper's setting; see `docs/methods/dcw.md` in anima_lora for why. |
| `0.0` | Disabled — no overhead, no extra hooks registered. |
| Positive values | Match the paper's direction. On Anima these *widen* the bias and over-smooth output. |

| `dcw_band_mask` | Behavior |
|---|---|
| `LL` (default) | Restrict correction to the Haar low-low subband. Strictly better than broadband on Anima — improves all four bands while broadband worsens the detail bands (LH/HL/HH). LL is the upstream causal lever; detail bands are downstream symptoms. |
| `all` | Paper-form broadband correction. Falls through to the cheap fused `add_` (no DWT round-trip). Pair with `dcw_lambda ≈ -0.010` if you switch to this. |
| `HH`, `LH+HL+HH` | Ablation modes. `HH`-only is empirically dead; `LH+HL+HH` pulled in for completeness. |

The schedule is fixed to `one_minus_sigma` (correction concentrates at low σ where Anima's bias is largest). Implementation is sampler-agnostic — DCW mutates the latent at the step boundary via a `CALC_COND_BATCH` wrapper plus a post-CFG capture hook, so it composes correctly with Euler / ER-SDE / DPM++ / etc., with CFG on or off, and stacks cleanly on top of Spectrum + mod guidance. The DWT/iDWT round-trip on `LL` mode is one pass over the latent (negligible vs the DiT forward).
