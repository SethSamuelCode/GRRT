# Auto-Exposure Tone Mapping

## Goal

Replace the fixed-exposure tone mapper with a two-pass auto-exposure system that adapts to the scene's HDR range, correctly exposing both the bright inner accretion disk and the dim star field.

## Problem

The disk emission at 10⁷K produces luminosity values ~10⁸× larger than the LUT ceiling. A fixed exposure either blows out the disk (both sides white, no Doppler beaming visible) or kills the stars and outer disk (too dark).

## Design

### Pass 1: Render to linear HDR

The renderer's OpenMP loop writes linear HDR values to the framebuffer — no tone mapping per pixel. This is also the correct output format for future Blender integration.

### Pass 2: Auto-expose and tone map

After all pixels are rendered, the `ToneMapper` scans the framebuffer to compute:

```
L_i = max(0, 0.2126 R + 0.7152 G + 0.0722 B)   (perceptual luminance, clamped non-negative)
L_avg = exp( (1/N) Σ log(δ + L_i) )              (log-average luminance, δ = 1e-6)
L_max = max(L_i)                                  (peak luminance)
```

**Edge case**: If `L_max == 0` (all-black frame), skip tone mapping entirely — leave framebuffer as-is.

Then applies tone mapping to each pixel:

```
exposure = key_value / L_avg                (key_value = 0.18, photographic middle gray)
L_scaled = L_pixel × exposure
L_white = L_max × exposure                  (auto white point from scene peak)
L_mapped = L_scaled × (1 + L_scaled / L_white²) / (1 + L_scaled)
L_mapped = clamp(L_mapped, 0, 1)           (guard against hot-pixel edge cases)
```

For each pixel, if `L_pixel > 0`: scale RGB by `L_mapped / L_pixel` to preserve chrominance, then apply sRGB gamma. If `L_pixel == 0`: output black (skip division).

`apply_all` is `const`-qualified — it modifies the framebuffer, not the ToneMapper object.

## Modified Files

| File | Change |
|------|--------|
| `include/grrt/render/tonemapper.h` | Replace `apply(Vec3)` with `apply_all(float* framebuffer, int width, int height)`. Add `key_value_` member (default 0.18). |
| `src/tonemapper.cpp` | Implement two-pass: compute stats, then tone-map in-place. |
| `include/grrt/render/renderer.h` | Remove `tonemapper_` from per-pixel use. Store reference, call `apply_all()` after render loop. |
| `src/renderer.cpp` | Write linear HDR in OpenMP loop. Call `tonemapper_.apply_all()` after. |

## Validation

- Stars visible alongside bright disk
- Doppler beaming clearly visible (left/right brightness asymmetry)
- Inner disk bright white, outer disk shows color gradient
- No all-black or all-white images
- Works for both Schwarzschild and Kerr at various spin values
