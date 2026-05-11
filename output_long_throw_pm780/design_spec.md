# PM780 Long-Throw Soft Flat-Top Shaper

## Optical Intent

- Source: PM780 fiber, MFD 5.30 um.
- Collimator: Thorlabs F220APC-780, f = 11.07 mm, NA = 0.26.
- Collimated Gaussian radius: w0 = 1.037 mm.
- Distance from collimator output plane to L1 flat face: 5.0 mm.
- Output profile: Fermi-Dirac soft flat-top.
- Nominal flat radius: 8.00 mm.
- Edge rolloff width: 1.25 mm.
- First required flat-top plane: 50 mm after L2.

## Lens Pair

- L1 diameter: 3.23 mm.
- L2 diameter: 23.13 mm.
- Center thicknesses: L1 2.00 mm, L2 2.00 mm.
- Lens-center separation: 120.0 mm.
- Assumed refractive index: 1.5200.
- OPL peak-to-peak: 0.000 waves.
- L1 STL: `lens1_pm780_long_throw.stl`.
- L2 STL: `lens2_pm780_long_throw.stl`.
- STL coordinate units: millimeters.

## Scalar Diffraction Proxy

- Core CV at 50 mm: 4.58%.
- Full nominal plateau CV at 50 mm: 15.20%.
- Scalar visualizer: `pm780_long_throw_visualizer.html`.

## 2tD GBD Propagation Check

- Simulation basis: 529 2tD gausslets on a polar_rings launch grid, fitted to the F220APC-780 collimated Gaussian launch.
- Radial launch coefficients enforced: True.
- Launch basis relative RMSE: 0.03%.
- Core CV at 50 mm: 3.09%.
- Full nominal plateau CV at 50 mm: 11.69%.
- 2tD visualizer: `pm780_long_throw_2td_gbd.html`.
- 2tD static plot: `pm780_long_throw_2td_gbd_summary.png`.
- 1D radial profile plot: `pm780_long_throw_2td_1d_profiles.html`.

## Notes

The scalar check models the intended second-lens output field with flat phase.
The 2tD GBD check propagates the fitted F220APC-780 Gaussian launch through the
actual tabulated lens surfaces in eigen_gbd.
The F220APC-780 package is represented as the post-collimator Gaussian launch
plane using the catalog focal length and NA; its internal prescription is not
traced in this design pass.
