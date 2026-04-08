# Gaussian-to-Flat-Top Beam Shaper

Two 2PP-printed aspheric lenses that convert a 780nm Gaussian beam into a collimated 15mm flat-top. Verified with Gaussian Beam Decomposition (eigen_gbd) wave optics.

```
[PM780 fiber] → [collimator] → [L1 3mm] --100mm-- [L2 15mm] → 15mm flat-top
                  buy this       print this          print this
```

## Setup

```bash
uv pip install -e "."
git submodule update --init  # for eigen_gbd
```

## Design and export

```python
from beamshaper.two_lens import design_two_lens_shaper
from beamshaper.export_stl import generate_lens_stl, write_stl_binary, write_sag_csv

result = design_two_lens_shaper(
    w0=1.0e-3, target_radius=7.5e-3, separation=100e-3,
    n_glass=1.52, t1=0.5e-3, t2=0.5e-3, aperture_radius=1.5e-3,
    n_points=10000)

for name, r, sag, t, ap in [
    ('lens1', result['r'], result['sag1'], 0.5e-3, result['r'][-1]),
    ('lens2', result['R'], result['sag2'], 0.5e-3, result['R'][-1])]:
    tris = generate_lens_stl(r, sag, t, ap, 200, 360)
    write_stl_binary(f'{name}.stl', tris * 1e3, name)
    write_sag_csv(f'{name}_sag.csv', r, sag, name)
```

## Wave-optics verification with eigen_gbd

The design is verified using [eigen_gbd](https://gitlab.com/mattjaffe/eigen_gbd) (included as a git submodule), which implements Gaussian Beam Decomposition — a wave-optics method that decomposes the field into Gaussian beamlets, propagates them through the optical system via ray tracing, and reconstructs the coherent field at the output. Unlike pure ray tracing, GBD captures diffraction and interference effects.

The GBD pipeline for our beam shaper:
1. Decompose the fiber/collimator Gaussian mode into ~51 beamlets (each with chief, waist, and divergence rays)
2. Propagate all beamlets through the two-lens system (refraction at each surface via Snell's law)
3. At the target plane, reconstruct the field: E(x) = sum of all beamlet fields (amplitude + phase)
4. |E(x)|^2 is the physical intensity profile — this is what a beam profiler would measure

This is what revealed that **equal OPL is critical**: ray tracing (which ignores phase) showed a flat-top for both collimated and diverging inputs, but GBD showed the diverging design produces a Gaussian field because the beamlets arrive with ~200 waves of phase error and interfere incoherently.

```python
import sys; sys.path.insert(0, 'eigen_gbd')
from beamshaper.gbd_simulator import run_gbd_simulation

sim = run_gbd_simulation(result, n_glass=1.52, source_distance=0,
    target_z_beyond=60e-3, w0_fiber=1.0e-3, wavelength=780e-9,
    n_gausslets=51, n_eval_pts=400)
# sim['I_normalized'] — intensity at target, should be flat-top with batman ears
# sim['E_field'] — complex field (amplitude and phase)
# sim['uniformity_cv'] — coefficient of variation in the flat-top region
```

The GBD simulator (`beamshaper/gbd_simulator.py`) defines a custom `TabulatedSurface` surface type for eigen_gbd that interpolates directly from our 10k-point sag tables, avoiding polynomial fit error.

## What's in output_collimated/

The recommended design. Uses a fiber collimator (Thorlabs F220APC-780) for a collimated Gaussian input, which gives OPL=0.000% and a proper flat-top confirmed by GBD.

| File | What |
|------|------|
| `lens1_redistributor.stl` | L1 solid, mm units, import into Think3D |
| `lens2_collimator.stl` | L2 solid, mm units |
| `lens1_sag.csv` / `lens2_sag.csv` | 10k-point surface profiles (r_mm, sag_um) |
| `collimated_beam_shaper_analysis.html` | Interactive 6-panel analysis plot |

## Lens specs

| | L1 (redistributor) | L2 (collimator) |
|---|---|---|
| Diameter | 3mm | 15mm |
| Center thickness | 0.5mm | 0.5mm |
| Edge thickness | 0.62mm | 0.98mm |
| Max sag | 123 um | 477 um |

Separation: 100mm. Mount in a cage system or lens tube with aspheric faces toward each other.

## Before printing

1. **Measure n at 780nm** for your resin. Design assumes 1.52. If different, rerun `design_two_lens_shaper(n_glass=YOUR_VALUE)`.
2. **Check the STL** in MeshLab/Blender. Units are mm.
3. Print flat face on glass substrate. Use adaptive resolution (fine on optical surface, coarse in bulk).

## Why collimated input matters

A diverging fiber source (no collimator) produces 0.125% OPL variation = ~200 waves of phase error. Ray tracing looks fine because it ignores phase, but GBD shows the field is Gaussian, not flat-top. With a collimator, OPL = 0.000% and GBD confirms a real flat-top with batman ears (2.4% CV at the target).

## How it works

1. Energy conservation maps each input radius r to an output radius R(r) = R_0 * sqrt(E(r)/E(a))
2. Snell's law at each aspheric surface determines the surface slope
3. OPL enforcement ensures all rays arrive in phase
4. Surface profiles are stored as raw sag tables (not polynomial fits) for 12x better collimation

The surfaces are computed by ODE integration (predictor-corrector, 10k points) and used directly — no polynomial fitting. This matters for fabrication on 2PP printers or diamond turning lathes that accept point clouds.

## Interactive UI

```bash
python run_app.py  # http://localhost:8050
```

Sliders for lens parameters, 2D/3D view toggle, gradient optimizer, analytical design button.
