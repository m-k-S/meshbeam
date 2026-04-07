# Beam Shaper: Gaussian-to-Flat-Top via Aspheric Lens Optimization

Designs and exports 2PP-printable plano-aspheric lens pairs that convert a Gaussian beam into a collimated flat-top beam profile. Verified by both ray optics and scalar wave optics (Collins integral).

## What this does

Takes a Gaussian laser beam (e.g. from a PM780 fiber) and produces a 15mm diameter flat-top (uniform intensity) beam that stays flat over several cm of propagation. The output is a pair of STL files you can print on an UpNano 2PP printer.

```
[PM780 fiber] --8mm-- [Lens 1 (3mm dia)] --100mm-- [Lens 2 (15mm dia)] → 15mm flat-top
                       2PP print                     2PP print
```

## Quick start

```bash
# Install (requires Python 3.10+, tested on 3.14)
uv pip install -e "."

# Run the design and export STLs
python -c "
from beamshaper.two_lens import design_two_lens_shaper
from beamshaper.tabulated_tracer import make_two_lens_tracer
from beamshaper.export_stl import generate_lens_stl, write_stl_binary, write_sag_csv

result = design_two_lens_shaper(
    w0=0.75e-3,              # effective beam radius at lens 1 (from PM780 at 8mm)
    target_radius=7.5e-3,    # 15mm diameter flat-top
    separation=100e-3,       # 100mm between lenses
    n_glass=1.52,            # IP-S resin at 780nm (VERIFY THIS)
    t1=0.5e-3, t2=0.5e-3,   # center thicknesses
    aperture_radius=1.5e-3,  # 3mm clear aperture lens 1
    n_points=10000,
    source_distance=8e-3,    # fiber tip 8mm from lens 1 flat face
)

# Export STLs (mm units, ready for Think3D import)
tris1 = generate_lens_stl(result['r'], result['sag1'], 0.5e-3, 1.5e-3, 200, 360)
write_stl_binary('output/lens1.stl', tris1 * 1e3, 'lens1')
tris2 = generate_lens_stl(result['R'], result['sag2'], 0.5e-3, result['R'][-1], 200, 360)
write_stl_binary('output/lens2.stl', tris2 * 1e3, 'lens2')

# Export sag tables as CSV (for direct import or diamond turning)
write_sag_csv('output/lens1_sag.csv', result['r'], result['sag1'], 'Lens 1')
write_sag_csv('output/lens2_sag.csv', result['R'], result['sag2'], 'Lens 2')
"

# Launch the interactive web UI
python run_app.py  # → http://localhost:8050
```

## Current design (PM780 fiber, no collimator)

### Input
- **Fiber**: 780PM (Thorlabs PM780-HP or equivalent), 0.12 NA, 5.3 um MFD
- **Wavelength**: 780 nm (monochromatic)
- **Fiber-to-Lens-1 distance**: 8mm (fiber tip to L1 flat face)

### Optical layout
| Element | Diameter | Center thick | Edge thick | Max sag |
|---------|----------|-------------|------------|---------|
| Lens 1 (redistributor) | 3mm | 0.50mm | 0.60mm | 105 um |
| Lens 2 (collimator) | 15mm | 0.50mm | 0.99mm | 493 um |
| Gap between lenses | - | 100mm | - | - |

### Performance (ray optics, 10k rays, tabulated surfaces)
| Distance from L2 exit | Uniformity (CV) |
|-----------------------|-----------------|
| 0mm | 5.0% |
| 20mm | 6.7% |
| 60mm (target start) | 8.3% |
| 100mm | 9.5% |
| 150mm | 10.9% |

Collimation: 353 urad mean divergence.

### Alternative design (with off-the-shelf collimator)
If you add a fiber collimator (e.g. Thorlabs F220APC-780, f=11mm) before the two printed lenses, the design becomes optically perfect:
- Remove `source_distance` parameter (collimated input)
- Set `w0=1.0e-3` (collimated beam from the collimator)
- Performance: **3.9% CV** (Poisson-limited), **1 urad** collimation, flat to >200mm

## How the design works

### The physics
1. **Energy conservation mapping**: For a Gaussian input I(r) = exp(-2r^2/w^2), the mapping R(r) that produces uniform output intensity on a disk of radius R_0 is:
   ```
   R(r) = R_0 * sqrt( E(r) / E(a) )
   where E(r) = 1 - exp(-2r^2/w^2), a = aperture radius
   ```
2. **Lens 1** refracts each ray at radius r so it arrives at radius R(r) on lens 2
3. **Lens 2** re-collimates the ray (output direction parallel to the optical axis)
4. **Equal OPL** (optical path length) ensures phase coherence → collimated output

### Surface computation
The aspheric profiles are computed by integrating the Snell's law slope equation:
```
dsag/dr = (n_air * sin_air - n_glass * sin_glass) / (n_air * cos_air - n_glass * cos_glass)
```
using a predictor-corrector (Heun's method) ODE integrator with 10,000 radial points. The surface is stored as a raw sag table — no polynomial fitting — because the tabulated surfaces give 12x better collimation than even 8th-order asphere polynomial fits.

### Key subtlety: beam sampling vs intensity
The Gaussian beam sampling in `mesh-optics-master/ray_sources.py` uses sigma = w/sqrt(2), which samples the **amplitude** distribution exp(-r^2/w^2), NOT the intensity exp(-2r^2/w^2). The beam shaper code uses sigma = w/2 (intensity-proportional sampling, each ray = equal power). The energy conservation mapping must use the matching CDF. Getting this wrong produces the "batman ears" artifact (edge intensity spikes).

## Before printing: verification checklist

### 1. Measure your resin's refractive index at 780nm
The design assumes **n = 1.52**. If your resin batch differs, rerun the design:
```python
result = design_two_lens_shaper(..., n_glass=YOUR_VALUE, ...)
```
Even a 0.01 change in n affects the surface profiles. IP-S typically ranges 1.50-1.55 depending on UV dose and post-curing.

### 2. Verify the STL in a mesh viewer
Open `lens1.stl` and `lens2.stl` in MeshLab, Blender, or Think3D. Check:
- Watertight (no holes)
- Smooth aspheric surface (no faceting artifacts)
- Dimensions match: L1 = 3mm dia x 0.6mm tall, L2 = 15mm dia x 1mm tall
- STL units are **millimeters**

### 3. Check the sag profiles
The CSV files have (r_mm, sag_um) with 10,000 points. Plot them to verify smooth monotonic profiles:
```python
import numpy as np, matplotlib.pyplot as plt
d = np.loadtxt('output/lens1_sag.csv', delimiter=',', comments='#')
plt.plot(d[:,0], d[:,1]); plt.xlabel('r (mm)'); plt.ylabel('sag (um)'); plt.show()
```

### 4. Print settings (UpNano NanoOne)
- **Resin**: IP-S or UpNano's optically transparent resin
- **Substrate**: Glass cover slip (#1.5, 170um thick)
- **Surface resolution**: Use adaptive resolution — fine (200nm) on the aspheric surface, coarser (1-5um) in the bulk interior
- **Surface roughness target**: <20nm RMS (achievable with 2PP)
- **Print orientation**: Flat face on the substrate (glass side = flat plano face)
- **Post-processing**: Standard development (PGMEA or as recommended for your resin). No additional polishing should be needed for 2PP-printed surfaces.

### 5. Alignment and mounting
- Mount both lenses in a 30mm cage system or SM1 lens tube
- **Lens 1**: flat face (glass substrate) toward the fiber, aspheric face toward lens 2
- **Lens 2**: aspheric face toward lens 1, flat face (glass substrate) toward the target
- **Spacing**: 100mm between lens centers (±0.5mm tolerance is fine — the flat-top is collimated, so spacing errors just shift the whole pattern laterally, they don't defocus)
- **Fiber positioning**: 8mm from fiber tip to lens 1 flat face. This distance matters more — ±0.5mm will change the effective beam size at lens 1.
- **Lateral alignment**: Center the fiber on the lens 1 optical axis within ~50um. Use a XY translation stage.

### 6. Testing the output
At the target plane (60mm+ from lens 2):
- Use a beam profiler (e.g. Thorlabs BP209) to measure the intensity profile
- Expected: 15mm diameter flat-top with <10% intensity variation
- Check at multiple z-positions (60mm, 80mm, 100mm) to verify collimation
- If you see "batman ears" (intensity spikes at the edge), the likely cause is a refractive index mismatch — remeasure n and redesign

## Project structure

```
beamshaping/
├── beamshaper/                    # Core library
│   ├── aspheric.py                # Aspheric sag equation + mesh generation
│   ├── analytical.py              # Single-lens analytical design (ODE)
│   ├── two_lens.py                # Two-lens design (coupled ODEs + OPL)
│   ├── jax_tracer.py              # Differentiable ray tracer (JAX, polynomial surfaces)
│   ├── tabulated_tracer.py        # Ray tracer using raw sag tables (no poly fit)
│   ├── optimizer.py               # Gradient descent optimizer (optax Adam)
│   ├── profiles.py                # Ray mapping + loss functions
│   ├── wave_optics.py             # 1D radial Collins integral propagation
│   ├── export_stl.py              # STL + CSV export for fabrication
│   └── visualize_system.py        # Plotly visualization of the full system
├── app/                           # Dash web UI
│   ├── main.py                    # App entry point
│   ├── layout.py                  # UI layout (sliders, buttons, plots)
│   ├── callbacks.py               # Interactive callbacks
│   ├── plot_2d.py / plot_3d.py    # Plotly figures
│   └── assets/style.css
├── mesh-optics-master/            # Original mesh-based ray tracer (reference)
├── output/                        # Generated fabrication files
│   ├── lens1_redistributor.stl    # Lens 1 solid (mm units)
│   ├── lens2_collimator.stl       # Lens 2 solid (mm units)
│   ├── lens1_sag.csv              # Surface profile (r_mm, sag_um)
│   ├── lens2_sag.csv              # Surface profile (r_mm, sag_um)
│   ├── beam_shaper_fiber_direct.html  # Interactive visualization
│   └── design_report.txt
├── run_app.py                     # Launch web UI
├── pyproject.toml
└── README.md
```

## Design approach options

| Approach | Uniformity | Depth of field | Complexity |
|----------|-----------|----------------|------------|
| Single lens, polynomial fit | 3% CV at one z | None (diverges immediately) | Low |
| Single lens, tabulated sag | 3% CV at one z | None | Low |
| Two lenses, polynomial fit | 6% CV | ~50mm at <10% CV | Medium |
| **Two lenses, tabulated sag** | **5% CV** | **>100mm at <10% CV** | Medium |
| Two lenses + collimator, tabulated | 4% CV | **>200mm at <5% CV** | Higher (3 elements) |

## Wave optics verification

A 1D radial Collins integral simulation (`wave_optics.py`) confirmed the single-lens flat-top at 7.9% CV, consistent with geometric optics plus Fresnel diffraction (central dip, edge ripple). The system Fresnel number is ~8000, so geometric optics dominates — diffraction effects are minor.

To run the wave optics sim:
```python
from beamshaper.analytical import design_beam_shaper
from beamshaper.aspheric import AsphericProfile
from beamshaper.wave_optics import simulate_beam_shaping

result = design_beam_shaper(w0=1e-3, target_radius=7.5e-3, target_z=0.25, n_alpha_coeffs=8)
profile = AsphericProfile(result['curvature'], result['conic'], tuple(result['alphas']), 1.5e-3)
sim = simulate_beam_shaping(profile, n_radial_in=80000, n_radial_out=400)
# sim['I_target'] is the radial intensity at the target
```

## Dependencies

```
jax, jaxlib     # Autodiff ray tracing
optax           # Gradient descent optimizer
plotly, dash    # Web UI + visualization
numpy, scipy    # Numerics
```

Optional (for mesh-optics-master): `numba, pyvista, matplotlib`

Install: `uv pip install -e ".[mesh]"`
