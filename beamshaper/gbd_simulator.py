"""
Gaussian Beam Decomposition (GBD) simulator for the beam shaper.

Uses eigen_gbd to:
1. Create the fiber output field as a Gaussian mode
2. Decompose into Gaussian beamlets
3. Propagate through the two-lens beam shaper
4. Recombine beamlets into a field at the target plane
5. Evaluate field amplitude uniformity

This provides a wave-optics-quality simulation that captures diffraction,
aberrations, and interference effects that pure ray tracing misses.
"""

import sys
import os
import numpy as np
from typing import Optional, List, Dict, Any

# Add eigen_gbd to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eigen_gbd'))

from eigen_gbd.units import mm, um, nm
from eigen_gbd.optical_setups import OpticalSetup1tD
from eigen_gbd.surfaces import Surface1tD
from eigen_gbd.surface_types import SurfaceType, Plane, Asphere
from eigen_gbd.gausslets import Gausslet1tD
from eigen_gbd.cavities import (
    build_gausslet_field_matrix,
    calculate_gausslet_amplitudes_from_efield_distribution,
)
from eigen_gbd.default_params import WVLN0, NIDX0


class TabulatedSurface(SurfaceType):
    """
    Custom surface type using interpolated sag tables.

    This preserves the exact surface profile from the ODE solver
    without polynomial fit error.
    """

    def __init__(self, r_table, sag_table, sgn=1, is_3d=False, surf_plot_style_opts=None, **kwargs):
        if surf_plot_style_opts is None:
            surf_plot_style_opts = {"lw": 2, "c": "xkcd:neon purple"}
        super().__init__(
            surf_fxn_kwargs={"r_table": r_table, "sag_table": sag_table, "sgn": sgn},
            is_3d=is_3d,
            surf_plot_style_opts=surf_plot_style_opts,
        )
        self.r_table = np.asarray(r_table)
        self.sag_table = np.asarray(sag_table)
        self.dsag_table = np.gradient(sag_table, r_table)
        self.sgn = sgn

    def surf_fxn(self):
        r_tab = self.r_table
        s_tab = self.sag_table
        sgn = self.sgn
        return lambda x: sgn * np.interp(np.abs(x), r_tab, s_tab)

    def d_surf_fxn(self):
        r_tab = self.r_table
        ds_tab = self.dsag_table
        sgn = self.sgn
        return lambda x: sgn * np.sign(x) * np.interp(np.abs(x), r_tab, ds_tab)


def create_optical_system(
    design_result: dict,
    n_glass: float = 1.52,
    source_distance: float = 8e-3,
    target_z_beyond: float = 60e-3,
    use_tabulated: bool = True,
) -> OpticalSetup1tD:
    """
    Create the full optical system for eigen_gbd propagation.

    Surfaces (in propagation order):
    0. Fiber output plane (z=0)
    1. Lens 1 flat face (z=source_distance)
    2. Lens 1 aspheric face (z=source_distance + t1)
    3. Lens 2 aspheric face (z=source_distance + separation - t2)
    4. Lens 2 flat face (z=source_distance + separation)
    5. Target plane (z=source_distance + separation + target_z_beyond)
    """
    t1 = design_result['t1']
    t2 = design_result['t2']
    sep = design_result['separation']
    ap1 = float(design_result['r'][-1])
    ap2 = float(design_result['R'][-1])

    z_fiber = 0.0
    z_l1_flat = source_distance
    z_l1_asph = source_distance + t1
    z_l2_asph = source_distance + sep - t2
    z_l2_flat = source_distance + sep
    z_target = source_distance + sep + target_z_beyond

    # Build surface kwargs for the aspheric faces
    if use_tabulated:
        surf1_cls = TabulatedSurface
        surf1_kw = {'r_table': design_result['r'], 'sag_table': design_result['sag1'], 'sgn': 1}
        surf2_cls = TabulatedSurface
        surf2_kw = {'r_table': design_result['R'], 'sag_table': design_result['sag2'], 'sgn': 1}
    else:
        c1 = design_result['coeffs1']
        c2 = design_result['coeffs2']
        R1 = 1.0 / c1['curvature'] if abs(c1['curvature']) > 1e-10 else 1e10
        R2 = 1.0 / c2['curvature'] if abs(c2['curvature']) > 1e-10 else 1e10
        surf1_cls = Asphere
        surf1_kw = {'R': R1, 'k': c1['conic'], 'As': list(c1['alphas']), 'sgn': 1}
        surf2_cls = Asphere
        surf2_kw = {'R': R2, 'k': c2['conic'], 'As': list(c2['alphas']), 'sgn': 1}

    surfaces = [
        # 0: Fiber output
        Surface1tD(z0=z_fiber, surface_type=Plane,
                   r_aperture=ap1 * 2, nIdx_before=1.0, nIdx_after=1.0),

        # 1: Lens 1 flat face (air → glass)
        Surface1tD(z0=z_l1_flat, surface_type=Plane,
                   r_aperture=ap1, nIdx_before=1.0, nIdx_after=n_glass),

        # 2: Lens 1 aspheric face (glass → air)
        Surface1tD(z0=z_l1_asph, surface_type=surf1_cls,
                   surface_kwargs=surf1_kw,
                   r_aperture=ap1, nIdx_before=n_glass, nIdx_after=1.0),

        # 3: Lens 2 aspheric face (air → glass)
        Surface1tD(z0=z_l2_asph, surface_type=surf2_cls,
                   surface_kwargs=surf2_kw,
                   r_aperture=ap2, nIdx_before=1.0, nIdx_after=n_glass),

        # 4: Lens 2 flat face (glass → air)
        Surface1tD(z0=z_l2_flat, surface_type=Plane,
                   r_aperture=ap2, nIdx_before=n_glass, nIdx_after=1.0),

        # 5: Target plane
        Surface1tD(z0=z_target, surface_type=Plane,
                   r_aperture=ap2 * 2, nIdx_before=1.0, nIdx_after=1.0),
    ]

    return OpticalSetup1tD(surfaces)


def run_gbd_simulation(
    design_result: dict,
    n_glass: float = 1.52,
    source_distance: float = 8e-3,
    target_z_beyond: float = 60e-3,
    w0_fiber: float = 2.65e-6,
    wavelength: float = 780e-9,
    n_gausslets: int = 51,
    overlap_factor: float = 1.75,
    n_eval_pts: int = 200,
    use_tabulated: bool = True,
) -> dict:
    """
    Run the full GBD simulation.

    Steps:
    1. Create the optical system
    2. Initialize gausslets at the fiber output with the fiber mode profile
    3. Propagate through the system
    4. Evaluate the field at the target plane

    Returns dict with:
        'r_target': radial positions at target
        'E_field': complex electric field at target
        'I_target': intensity |E|^2
        'I_normalized': intensity normalized to peak
        'uniformity_cv': coefficient of variation in the flat-top region
        'gausslets': propagated gausslet objects
        'optical_setup': the optical system
    """
    ap2 = float(design_result['R'][-1])
    target_radius = ap2

    # 1. Create optical system
    print("  Creating optical system...")
    optical_setup = create_optical_system(
        design_result, n_glass, source_distance, target_z_beyond, use_tabulated)

    # 2. Initialize gausslets at fiber output (surface 0)
    # The fiber mode is a Gaussian with w0 = w0_fiber = MFD/2
    # We set the gausslet waist to be appropriate for decomposition
    print(f"  Initializing {n_gausslets} gausslets...")

    # The fiber mode extends over ~3*w0 ≈ 8um, very small compared to
    # the downstream optics. The gausslet waist should be comparable to w0_fiber.
    # Use a beam range that covers the fiber mode.
    fiber_rng = 4 * w0_fiber  # ±4*w0 covers >99.99% of energy

    gausslets = optical_setup.initialize_Gausslets(
        surf_idx=0,
        n_gausslets=n_gausslets,
        overlap_factor=overlap_factor,
        wvln=wavelength,
        rng=fiber_rng,
    )

    # 3. Set gausslet amplitudes to match the fiber Gaussian mode
    # The fiber output field: E(x) = exp(-x^2/w0^2)
    # Decompose this onto the gausslet basis
    n_decomp = max(n_gausslets * 3, 200)
    x_decomp = np.linspace(-fiber_rng, fiber_rng, n_decomp)
    pts_decomp = np.column_stack([x_decomp, np.full(n_decomp, 0.0)])

    # Fiber mode amplitude
    E_fiber = np.exp(-x_decomp ** 2 / w0_fiber ** 2)

    # Build the gausslet field matrix at the fiber plane
    G_fiber = build_gausslet_field_matrix(pts_decomp, gausslets, propLegIdx=0)

    # Solve for amplitudes
    c_fiber = calculate_gausslet_amplitudes_from_efield_distribution(
        pts_decomp, E_fiber, gausslets, G=G_fiber)

    # Apply amplitudes to gausslets
    for j, glet in enumerate(gausslets):
        glet.ampl = c_fiber[j]

    # Verify decomposition
    E_reconstructed = G_fiber @ c_fiber
    decomp_error = np.sqrt(np.mean(np.abs(E_reconstructed - E_fiber) ** 2)) / np.max(np.abs(E_fiber))
    print(f"  Fiber mode decomposition error: {decomp_error:.2e}")

    # 4. Propagate through the system surface by surface
    print("  Propagating through optical system...")
    n_surfaces = len(optical_setup.surfaces)
    for dest in range(1, n_surfaces):
        gausslets = optical_setup.prop_Gausslets_to_surface(gausslets, dest_surf_idx=dest)
    n_segs = len(gausslets[0].chief_ray_paths[0].ray_segments)
    print(f"  Propagated through {n_surfaces} surfaces ({n_segs} ray segments each)")

    # Compute Gouy phases (needed for field evaluation)
    for glet in gausslets:
        glet.compute_gouy_phases()

    # 5. Evaluate field at target plane
    print("  Evaluating field at target plane...")
    x_target = np.linspace(-target_radius * 1.1, target_radius * 1.1, n_eval_pts)
    z_target = optical_setup.surfaces[-1].z0
    pts_target = np.column_stack([x_target, np.full(n_eval_pts, z_target)])

    # Last propagation leg (after last surface)
    propLegIdx = n_segs - 1

    G_target = build_gausslet_field_matrix(pts_target, gausslets, propLegIdx=propLegIdx)
    E_target = G_target @ c_fiber

    I_target = np.abs(E_target) ** 2
    I_normalized = I_target / np.max(I_target) if np.max(I_target) > 0 else I_target

    # 6. Compute uniformity in flat-top region
    mask_flat = (np.abs(x_target) > target_radius * 0.05) & (np.abs(x_target) < target_radius * 0.85)
    I_flat = I_normalized[mask_flat]
    cv = np.std(I_flat) / np.mean(I_flat) if np.mean(I_flat) > 0 else 999

    print(f"  Uniformity in flat-top region: {cv*100:.1f}% CV")

    return {
        'x_target': x_target,
        'E_field': E_target,
        'I_target': I_target,
        'I_normalized': I_normalized,
        'uniformity_cv': cv,
        'gausslets': gausslets,
        'optical_setup': optical_setup,
        'c_fiber': c_fiber,
        'decomp_error': decomp_error,
    }


def gbd_loss_function(
    asphere_params: np.ndarray,
    design_kwargs: dict,
    gbd_kwargs: dict,
) -> float:
    """
    Loss function for optimization: run GBD simulation and return
    the non-uniformity of the output field.

    Args:
        asphere_params: [c1, k1, a1_0, a1_1, ..., c2, k2, a2_0, ...]
        design_kwargs: kwargs for design_two_lens_shaper
        gbd_kwargs: kwargs for run_gbd_simulation

    Returns:
        loss: coefficient of variation of the intensity in the flat-top region
    """
    from beamshaper.two_lens import design_two_lens_shaper

    n_alpha = design_kwargs.get('n_alpha_coeffs', 4)

    # Unpack parameters — not used directly for tabulated design,
    # but we can use them to perturb the analytical design
    # For now, just use the standard analytical design
    result = design_two_lens_shaper(**design_kwargs)

    try:
        sim = run_gbd_simulation(result, **gbd_kwargs)
        return sim['uniformity_cv']
    except Exception as e:
        print(f"  GBD simulation failed: {e}")
        return 999.0
