"""
GBD-based optimization of the beam shaper lens surfaces.

Uses the eigen_gbd field amplitude as the forward model and optimizes
lens surface parameters (R, k, polynomial coefficients) to minimize
deviation from a rectangular (step-function) target profile.

The optimization is derivative-free (Powell's method) since eigen_gbd
is not differentiable. Each forward evaluation takes ~2s.
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eigen_gbd'))

from eigen_gbd.units import mm, um, nm
from eigen_gbd.optical_setups import OpticalSetup1tD
from eigen_gbd.surfaces import Surface1tD
from eigen_gbd.surface_types import Plane, Asphere
from eigen_gbd.cavities import (
    build_gausslet_field_matrix,
    calculate_gausslet_amplitudes_from_efield_distribution,
)
from eigen_gbd.default_params import WVLN0


def _setup_and_evaluate(
    R1, k1, As1, R2, k2, As2,
    separation, t1, t2, ap1, ap2, n_glass,
    source_distance, w0_fiber, wavelength,
    n_gausslets, n_eval_pts,
):
    """
    Set up optical system with polynomial aspheric surfaces,
    propagate gausslets, evaluate field at target.
    """
    z_l1_flat = source_distance
    z_l1_asph = source_distance + t1
    z_l2_asph = source_distance + separation - t2
    z_l2_flat = source_distance + separation
    # Place target right at L2 exit + 60mm
    z_target = z_l2_flat + 60e-3

    surfaces = [
        Surface1tD(z0=0.0, surface_type=Plane, r_aperture=ap1*2),
        Surface1tD(z0=z_l1_flat, surface_type=Plane,
                   r_aperture=ap1, nIdx_before=1.0, nIdx_after=n_glass),
        Surface1tD(z0=z_l1_asph, surface_type=Asphere,
                   surface_kwargs={'R': R1, 'k': k1, 'As': list(As1), 'sgn': 1},
                   r_aperture=ap1, nIdx_before=n_glass, nIdx_after=1.0),
        Surface1tD(z0=z_l2_asph, surface_type=Asphere,
                   surface_kwargs={'R': R2, 'k': k2, 'As': list(As2), 'sgn': 1},
                   r_aperture=ap2, nIdx_before=1.0, nIdx_after=n_glass),
        Surface1tD(z0=z_l2_flat, surface_type=Plane,
                   r_aperture=ap2, nIdx_before=n_glass, nIdx_after=1.0),
        Surface1tD(z0=z_target, surface_type=Plane, r_aperture=ap2*2),
    ]

    optical_setup = OpticalSetup1tD(surfaces)

    # Initialize gausslets at fiber
    fiber_rng = 4 * w0_fiber
    gausslets = optical_setup.initialize_Gausslets(
        surf_idx=0, n_gausslets=n_gausslets,
        overlap_factor=1.75, wvln=wavelength, rng=fiber_rng)

    # Decompose fiber mode
    n_decomp = max(n_gausslets * 3, 200)
    x_decomp = np.linspace(-fiber_rng, fiber_rng, n_decomp)
    pts_decomp = np.column_stack([x_decomp, np.zeros(n_decomp)])
    E_fiber = np.exp(-x_decomp**2 / w0_fiber**2)
    G_fiber = build_gausslet_field_matrix(pts_decomp, gausslets, propLegIdx=0)
    c = calculate_gausslet_amplitudes_from_efield_distribution(
        pts_decomp, E_fiber, gausslets, G=G_fiber)
    for j, glet in enumerate(gausslets):
        glet.ampl = c[j]

    # Propagate
    for dest in range(1, len(surfaces)):
        gausslets = optical_setup.prop_Gausslets_to_surface(gausslets, dest_surf_idx=dest)
    for glet in gausslets:
        glet.compute_gouy_phases()

    # Evaluate field at target
    x_target = np.linspace(-ap2*1.1, ap2*1.1, n_eval_pts)
    pts_target = np.column_stack([x_target, np.full(n_eval_pts, z_target)])
    n_segs = len(gausslets[0].chief_ray_paths[0].ray_segments)
    G_target = build_gausslet_field_matrix(pts_target, gausslets, propLegIdx=n_segs-1)
    E_target = G_target @ c

    return x_target, E_target


def step_function_loss(x, E, target_radius, edge_width=None):
    """
    Loss comparing |E|^2 to a step function (ideal flat-top).

    target: I(x) = 1 for |x| < target_radius, 0 for |x| > target_radius
    (smoothed with a tanh edge for differentiability)
    """
    if edge_width is None:
        edge_width = target_radius * 0.03  # 3% edge width

    I = np.abs(E)**2
    # Normalize so the mean intensity in the central 50% equals 1
    central = np.abs(x) < target_radius * 0.5
    if np.sum(central) > 0 and np.mean(I[central]) > 0:
        I = I / np.mean(I[central])

    # Target: smooth step function
    target = 0.5 * (1 - np.tanh((np.abs(x) - target_radius) / edge_width))

    # MSE weighted more heavily in the flat region and at the edge
    weight = np.ones_like(x)
    weight[np.abs(x) < target_radius * 0.9] = 2.0   # flat region matters most
    edge_mask = (np.abs(x) > target_radius * 0.8) & (np.abs(x) < target_radius * 1.2)
    weight[edge_mask] = 3.0  # edge sharpness matters a lot

    mse = np.sum(weight * (I - target)**2) / np.sum(weight)
    return float(mse)


def _setup_and_evaluate_tabulated(
    sag1, sag2, r1_table, r2_table,
    separation, t1, t2, ap1, ap2, n_glass,
    source_distance, w0_fiber, wavelength,
    n_gausslets, n_eval_pts,
    target_z_beyond=60e-3,
):
    """
    Evaluate GBD using tabulated sag profiles (TabulatedSurface).
    More stable than Asphere since the sag is always valid.
    """
    from beamshaper.gbd_simulator import TabulatedSurface

    z_l1_flat = source_distance
    z_l1_asph = source_distance + t1
    z_l2_asph = source_distance + separation - t2
    z_l2_flat = source_distance + separation
    z_target = z_l2_flat + target_z_beyond

    surfaces = [
        Surface1tD(z0=0.0, surface_type=Plane, r_aperture=ap1 * 2),
        Surface1tD(z0=z_l1_flat, surface_type=Plane,
                   r_aperture=ap1, nIdx_before=1.0, nIdx_after=n_glass),
        Surface1tD(z0=z_l1_asph, surface_type=TabulatedSurface,
                   surface_kwargs={'r_table': r1_table, 'sag_table': sag1, 'sgn': 1},
                   r_aperture=ap1, nIdx_before=n_glass, nIdx_after=1.0),
        Surface1tD(z0=z_l2_asph, surface_type=TabulatedSurface,
                   surface_kwargs={'r_table': r2_table, 'sag_table': sag2, 'sgn': 1},
                   r_aperture=ap2, nIdx_before=1.0, nIdx_after=n_glass),
        Surface1tD(z0=z_l2_flat, surface_type=Plane,
                   r_aperture=ap2, nIdx_before=n_glass, nIdx_after=1.0),
        Surface1tD(z0=z_target, surface_type=Plane, r_aperture=ap2 * 2),
    ]

    optical_setup = OpticalSetup1tD(surfaces)

    fiber_rng = 4 * w0_fiber
    gausslets = optical_setup.initialize_Gausslets(
        surf_idx=0, n_gausslets=n_gausslets,
        overlap_factor=1.75, wvln=wavelength, rng=fiber_rng)

    n_decomp = max(n_gausslets * 3, 200)
    x_decomp = np.linspace(-fiber_rng, fiber_rng, n_decomp)
    pts_decomp = np.column_stack([x_decomp, np.zeros(n_decomp)])
    E_fiber = np.exp(-x_decomp ** 2 / w0_fiber ** 2)
    G_fiber = build_gausslet_field_matrix(pts_decomp, gausslets, propLegIdx=0)
    c = calculate_gausslet_amplitudes_from_efield_distribution(
        pts_decomp, E_fiber, gausslets, G=G_fiber)
    for j, glet in enumerate(gausslets):
        glet.ampl = c[j]

    for dest in range(1, len(surfaces)):
        gausslets = optical_setup.prop_Gausslets_to_surface(gausslets, dest_surf_idx=dest)
    for glet in gausslets:
        glet.compute_gouy_phases()

    x_target = np.linspace(-ap2 * 1.1, ap2 * 1.1, n_eval_pts)
    pts_target = np.column_stack([x_target, np.full(n_eval_pts, z_target)])
    n_segs = len(gausslets[0].chief_ray_paths[0].ray_segments)
    G_target = build_gausslet_field_matrix(pts_target, gausslets, propLegIdx=n_segs - 1)
    E_target = G_target @ c

    return x_target, E_target


def optimize_beam_shaper_gbd(
    design_result: dict,
    n_glass: float = 1.52,
    source_distance: float = 8e-3,
    w0_fiber: float = 2.65e-6,
    wavelength: float = 780e-9,
    target_radius: float = 7.5e-3,
    n_gausslets: int = 41,
    n_eval_pts: int = 150,
    max_iter: int = 150,
    n_perturb_coeffs: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Optimize the beam shaper using GBD field vs step-function target.

    Strategy: keep the analytical sag tables as baseline, optimize
    additive even-polynomial perturbations on each surface:
        sag_opt(r) = sag_analytical(r) + Σ p_j * r^(2j+2)
    This is stable because the perturbations are small corrections.
    """
    r1_table = design_result['r']
    r2_table = design_result['R']
    sag1_base = design_result['sag1'].copy()
    sag2_base = design_result['sag2'].copy()
    separation = design_result['separation']
    t1 = design_result['t1']
    t2 = design_result['t2']
    ap1 = float(r1_table[-1])
    ap2 = float(r2_table[-1])

    nc = n_perturb_coeffs
    # Start with zero perturbation
    x0 = np.zeros(2 * nc)

    iteration_data = {'iter': 0, 'best_loss': np.inf, 'best_x': x0.copy(), 'history': []}

    def apply_perturbation(params):
        """Add polynomial perturbation to baseline sag."""
        p1_coeffs = params[:nc]
        p2_coeffs = params[nc:]

        sag1 = sag1_base.copy()
        for j, pj in enumerate(p1_coeffs):
            sag1 += pj * r1_table ** (2 * j + 2)

        sag2 = sag2_base.copy()
        for j, pj in enumerate(p2_coeffs):
            sag2 += pj * r2_table ** (2 * j + 2)

        return sag1, sag2

    def objective(params):
        sag1, sag2 = apply_perturbation(params)

        try:
            x, E = _setup_and_evaluate_tabulated(
                sag1, sag2, r1_table, r2_table,
                separation, t1, t2, ap1, ap2, n_glass,
                source_distance, w0_fiber, wavelength,
                n_gausslets, n_eval_pts)

            loss = step_function_loss(x, E, target_radius)
            if np.isnan(loss) or np.isinf(loss):
                loss = 100.0
        except Exception as e:
            loss = 100.0

        iteration_data['iter'] += 1
        iteration_data['history'].append(loss)
        if loss < iteration_data['best_loss']:
            iteration_data['best_loss'] = loss
            iteration_data['best_x'] = params.copy()

        if verbose and iteration_data['iter'] % 10 == 0:
            print(f"  [{iteration_data['iter']:4d}] loss={loss:.6f} (best={iteration_data['best_loss']:.6f})")

        return loss

    if verbose:
        print(f"GBD optimization: {2*nc} params, {max_iter} max iterations")
        loss0 = objective(x0)
        print(f"  Initial loss (vs step function): {loss0:.6f}")

    result = minimize(objective, x0, method='Powell',
                      options={'maxiter': max_iter, 'maxfev': max_iter * 10,
                               'ftol': 1e-7})

    # Use best parameters found (not necessarily the final iterate)
    best_params = iteration_data['best_x']
    sag1_opt, sag2_opt = apply_perturbation(best_params)

    # Final high-res evaluation
    x_final, E_final = _setup_and_evaluate_tabulated(
        sag1_opt, sag2_opt, r1_table, r2_table,
        separation, t1, t2, ap1, ap2, n_glass,
        source_distance, w0_fiber, wavelength,
        n_gausslets, n_eval_pts * 2)

    final_loss = step_function_loss(x_final, E_final, target_radius)

    if verbose:
        print(f"\nDone ({result.nfev} evals). Final loss: {final_loss:.6f}")
        print(f"  Perturbation norms: L1={np.max(np.abs(best_params[:nc])):.2e}, L2={np.max(np.abs(best_params[nc:])):.2e}")

    return {
        'x_target': x_final,
        'E_target': E_final,
        'sag1': sag1_opt,
        'sag2': sag2_opt,
        'r1_table': r1_table,
        'r2_table': r2_table,
        'perturbation_coeffs': best_params,
        'loss_history': iteration_data['history'],
        'final_loss': final_loss,
        'separation': separation,
        't1': t1, 't2': t2,
        'ap1': ap1, 'ap2': ap2,
    }
