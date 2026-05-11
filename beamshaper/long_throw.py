"""
Long-throw PM780 flat-top beam shaper design and visualizer.

This module builds a new two-lens soft-edge design for a PM780 fiber launched
through a Thorlabs F220APC-780 collimator, then evaluates the lens pair with
both a scalar diffraction proxy and a lens-by-lens 2tD Gaussian beam
decomposition (GBD) propagation through the tabulated surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import j0

from beamshaper.export_stl import generate_lens_stl, write_sag_csv, write_stl_binary
from beamshaper.two_lens import design_two_lens_shaper


@dataclass(frozen=True)
class LongThrowDefaults:
    wavelength: float = 780e-9
    fiber_mfd: float = 5.3e-6
    collimator_model: str = "Thorlabs F220APC-780"
    collimator_focal_length: float = 11.07e-3
    collimator_na: float = 0.26
    collimator_to_l1: float = 5.0e-3
    n_glass: float = 1.52
    input_aperture_radius: float = 1.70e-3
    flat_radius: float = 8.0e-3
    edge_width: float = 1.25e-3
    lens_separation: float = 120e-3
    lens1_thickness: float = 2.0e-3
    lens2_thickness: float = 2.0e-3
    first_eval_distance: float = 50e-3


DEFAULTS = LongThrowDefaults()


def pm780_collimated_radius(
    wavelength: float = DEFAULTS.wavelength,
    fiber_mfd: float = DEFAULTS.fiber_mfd,
    collimator_focal_length: float = DEFAULTS.collimator_focal_length,
) -> float:
    """
    Return the 1/e field radius after a diffraction-limited collimator.

    PM780-HP is commonly specified around 5.3 um MFD at 780 nm. For a Gaussian
    mode at the front focal plane of a thin collimator, the collimated beam
    radius is w = lambda*f/(pi*w_fiber).
    """
    fiber_mode_radius = 0.5 * fiber_mfd
    return wavelength * collimator_focal_length / (np.pi * fiber_mode_radius)


def _ensure_eigen_gbd_path() -> None:
    """Allow direct module execution without installing the submodule package."""
    repo_root = Path(__file__).resolve().parents[1]
    eigen_path = repo_root / "eigen_gbd"
    if str(eigen_path) not in sys.path:
        sys.path.insert(0, str(eigen_path))


def surface_type_2td():
    _ensure_eigen_gbd_path()
    from eigen_gbd.surface_types import SurfaceType

    class _TabulatedSurface2tD(SurfaceType):
        def __init__(self, r_table, sag_table, sgn=-1, is_3d=True, surf_plot_style_opts=None, **kwargs):
            if surf_plot_style_opts is None:
                surf_plot_style_opts = {"color": 0x3A86FF}
            super().__init__(
                surf_fxn_kwargs={"r_table": r_table, "sag_table": sag_table, "sgn": sgn},
                is_3d=is_3d,
                surf_plot_style_opts=surf_plot_style_opts,
            )
            self.r_table = np.asarray(r_table, dtype=float)
            self.sag_table = np.asarray(sag_table, dtype=float)
            self.dsag_table = np.gradient(self.sag_table, self.r_table)
            self.sgn = float(sgn)

        def _sag(self, x, y):
            r = np.hypot(x, y)
            return np.interp(r, self.r_table, self.sag_table, left=self.sag_table[0], right=self.sag_table[-1])

        def _dsag(self, x, y):
            r = np.hypot(x, y)
            return np.interp(r, self.r_table, self.dsag_table, left=self.dsag_table[0], right=self.dsag_table[-1])

        def surf_fxn(self):
            return lambda x, y: self.sgn * self._sag(x, y)

        def d_surf_fxn(self):
            def dfdx(x, y):
                r = np.hypot(x, y)
                if r < 1e-18:
                    return 0.0
                return self.sgn * self._dsag(x, y) * x / r

            def dfdy(x, y):
                r = np.hypot(x, y)
                if r < 1e-18:
                    return 0.0
                return self.sgn * self._dsag(x, y) * y / r

            return [dfdx, dfdy]

    return _TabulatedSurface2tD


def soft_top_intensity(r: np.ndarray, flat_radius: float, edge_width: float) -> np.ndarray:
    """Fermi-Dirac soft flat-top intensity profile."""
    x = np.clip((np.asarray(r) - flat_radius) / edge_width, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(x))


def design_pm780_long_throw_shaper(
    *,
    wavelength: float = DEFAULTS.wavelength,
    fiber_mfd: float = DEFAULTS.fiber_mfd,
    collimator_focal_length: float = DEFAULTS.collimator_focal_length,
    input_w0: float | None = None,
    flat_radius: float = DEFAULTS.flat_radius,
    edge_width: float = DEFAULTS.edge_width,
    separation: float = DEFAULTS.lens_separation,
    n_glass: float = DEFAULTS.n_glass,
    t1: float = DEFAULTS.lens1_thickness,
    t2: float = DEFAULTS.lens2_thickness,
    input_aperture_radius: float = DEFAULTS.input_aperture_radius,
    n_points: int = 10_000,
) -> dict:
    """
    Create a soft-edge long-throw two-lens design for a collimated PM780 beam.

    Defaults keep the second lens within a 1 inch optic envelope while using a
    deliberately relaxed edge: an 8 mm nominal flat radius with a 1.25 mm
    Fermi rolloff, giving an output clear radius near 11.5-12 mm.
    """
    if input_w0 is None:
        input_w0 = pm780_collimated_radius(
            wavelength=wavelength,
            fiber_mfd=fiber_mfd,
            collimator_focal_length=collimator_focal_length,
        )

    result = design_two_lens_shaper(
        w0=input_w0,
        target_radius=flat_radius,
        separation=separation,
        n_glass=n_glass,
        t1=t1,
        t2=t2,
        aperture_radius=input_aperture_radius,
        n_points=n_points,
        profile_type="fermi_dirac",
        edge_width=edge_width,
        source_distance=None,
    )

    result.update({
        "wavelength": wavelength,
        "fiber_mfd": fiber_mfd,
        "collimator_model": DEFAULTS.collimator_model,
        "collimator_focal_length": collimator_focal_length,
        "collimator_na": DEFAULTS.collimator_na,
        "collimator_to_l1": DEFAULTS.collimator_to_l1,
        "input_w0": input_w0,
        "input_aperture_radius": input_aperture_radius,
        "flat_radius": flat_radius,
        "edge_width": edge_width,
        "output_radius": float(result["R"][-1]),
        "profile_type": "fermi_dirac",
    })
    return result


def _trapz_weights(r: np.ndarray) -> np.ndarray:
    weights = np.empty_like(r)
    weights[1:-1] = 0.5 * (r[2:] - r[:-2])
    weights[0] = 0.5 * (r[1] - r[0])
    weights[-1] = 0.5 * (r[-1] - r[-2])
    return weights


def propagate_radial_field(
    r_in: np.ndarray,
    e_in: np.ndarray,
    r_out: np.ndarray,
    distance: float,
    wavelength: float,
    chunk_size: int = 64,
) -> np.ndarray:
    """
    Propagate a cylindrically symmetric scalar field by the Fresnel-Hankel integral.

    The absolute prefactor is retained, but downstream metrics normalize each
    plane, so only the relative radial shape matters.
    """
    if distance <= 0:
        return np.interp(r_out, r_in, e_in, left=e_in[0], right=0.0).astype(complex)

    k = 2.0 * np.pi / wavelength
    weights = _trapz_weights(r_in)
    base = e_in * np.exp(1j * k * r_in ** 2 / (2.0 * distance)) * r_in * weights
    e_out = np.empty_like(r_out, dtype=complex)

    for start in range(0, len(r_out), chunk_size):
        stop = min(start + chunk_size, len(r_out))
        rr = r_out[start:stop]
        kernel = j0((k / distance) * r_in[:, None] * rr[None, :])
        e_out[start:stop] = np.sum(base[:, None] * kernel, axis=0)

    e_out *= (2.0 * np.pi / (1j * wavelength * distance))
    e_out *= np.exp(1j * k * r_out ** 2 / (2.0 * distance))
    return e_out


def simulate_long_throw_performance(
    design: dict,
    distances: Iterable[float] | None = None,
    *,
    n_radial_in: int = 8192,
    n_radial_out: int = 420,
) -> dict:
    """
    Evaluate soft-top diffraction after the second lens over several distances.

    Distances are measured from the second-lens flat exit face. The default
    range starts at 50 mm because that is the user's first required flat-top
    plane and extends to 1 m for depth-of-field inspection.
    """
    if distances is None:
        distances = np.array([0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00])
    distances = np.asarray(list(distances), dtype=float)

    wavelength = float(design["wavelength"])
    flat_radius = float(design["flat_radius"])
    edge_width = float(design["edge_width"])
    output_radius = float(design["output_radius"])

    r_in = np.linspace(0.0, output_radius, n_radial_in)
    target_i_in = soft_top_intensity(r_in, flat_radius, edge_width)
    e_in = np.sqrt(target_i_in).astype(complex)

    r_out = np.linspace(0.0, output_radius * 1.15, n_radial_out)
    target_i_out = soft_top_intensity(r_out, flat_radius, edge_width)

    intensities = np.empty((len(distances), len(r_out)))
    fields = np.empty((len(distances), len(r_out)), dtype=complex)
    for i, distance in enumerate(distances):
        field = propagate_radial_field(r_in, e_in, r_out, float(distance), wavelength)
        intensity = np.abs(field) ** 2
        peak = float(np.max(intensity))
        if peak > 0:
            intensity = intensity / peak
        fields[i] = field
        intensities[i] = intensity

    core_mask = r_out <= 0.75 * flat_radius
    plateau_mask = r_out <= flat_radius
    cv_core = np.std(intensities[:, core_mask], axis=1) / (
        np.mean(intensities[:, core_mask], axis=1) + 1e-30
    )
    cv_plateau = np.std(intensities[:, plateau_mask], axis=1) / (
        np.mean(intensities[:, plateau_mask], axis=1) + 1e-30
    )
    rms_target = np.sqrt(np.mean((intensities - target_i_out[None, :]) ** 2, axis=1))

    return {
        "distances": distances,
        "r_in": r_in,
        "r_out": r_out,
        "E_fields": fields,
        "I_profiles": intensities,
        "I_target": target_i_out,
        "cv_core": cv_core,
        "cv_plateau": cv_plateau,
        "rms_target": rms_target,
    }


def export_lens_stls(
    design: dict,
    output_dir: str | Path,
    *,
    radial_segments: int = 240,
    azimuth_segments: int = 360,
) -> dict[str, Path]:
    """Export individual watertight STL meshes for the two plano-aspheric lenses."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lens1_tris = generate_lens_stl(
        design["r"],
        design["sag1"],
        float(design["t1"]),
        float(design["r"][-1]),
        radial_segments=radial_segments,
        azimuth_segments=azimuth_segments,
    )
    lens2_tris = generate_lens_stl(
        design["R"],
        design["sag2"],
        float(design["t2"]),
        float(design["R"][-1]),
        radial_segments=radial_segments,
        azimuth_segments=azimuth_segments,
    )

    lens1_path = out / "lens1_pm780_long_throw.stl"
    lens2_path = out / "lens2_pm780_long_throw.stl"
    write_stl_binary(lens1_path, lens1_tris * 1e3, solid_name="PM780_long_throw_lens1_mm")
    write_stl_binary(lens2_path, lens2_tris * 1e3, solid_name="PM780_long_throw_lens2_mm")
    return {"lens1_stl": lens1_path, "lens2_stl": lens2_path}


def _disc_points(radius: float, side: int, z: float) -> np.ndarray:
    xs = np.linspace(-radius, radius, side)
    ys = np.linspace(-radius, radius, side)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    rr = np.hypot(xx, yy)
    mask = rr <= radius
    return np.column_stack([xx[mask], yy[mask], np.full(np.count_nonzero(mask), z)])


def _cartesian_plane_points(radius: float, side: int, z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-radius, radius, side)
    ys = np.linspace(-radius, radius, side)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.full(side * side, z)])
    return pts, xx, yy


def _polar_ring_xy(radius: float, n_rings: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return rotationally balanced launch points and ring group IDs.

    Ring j has 8*j angular samples, which cancels the low-order angular modes
    that showed up as ellipticity in sparse square/fibonacci launch bases.
    """
    xy = [(0.0, 0.0)]
    groups = [0]
    for ring in range(1, n_rings + 1):
        r = radius * ring / n_rings
        n_theta = 8 * ring
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        xy.extend(zip(r * np.cos(theta), r * np.sin(theta)))
        groups.extend([ring] * n_theta)
    return np.asarray(xy, dtype=float), np.asarray(groups, dtype=int)


def _initialize_balanced_2td_gausslets(
    setup,
    *,
    wavelength: float,
    launch_radius: float,
    overlap_factor: float,
    n_rings: int,
):
    """Initialize a rotationally balanced polar-ring 2tD gausslet basis."""
    _ensure_eigen_gbd_path()
    from eigen_gbd.gausslets import Gausslet2tD

    surface = setup.surfaces[0]
    xy, groups = _polar_ring_xy(launch_radius, n_rings)
    w0_g = overlap_factor * launch_radius / n_rings
    gausslets = []
    for x, y in xy:
        z_rel = surface.surface_type.surf_fxn()(x, y)
        p0 = surface.origin + np.array([x, y, z_rel])
        vhat = surface.surface_type.surface_normal_fxn(x, y)
        gausslets.append(
            Gausslet2tD(
                p0=p0,
                v0=vhat,
                w0=w0_g,
                wvln=wavelength,
                nIdx0=surface.nIdx_after,
                surface_type=surface.surface_type,
                initialize_on_curved_surface=False,
                surface_z0=surface.z0,
            )
        )
    return gausslets, groups


def _radial_average_map(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    intensity_map: np.ndarray,
    *,
    r_max: float,
    n_bins: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    rr = np.hypot(x_grid.ravel(), y_grid.ravel())
    ii = intensity_map.ravel()
    bins = np.linspace(0.0, r_max, n_bins + 1)
    sums, _ = np.histogram(rr, bins=bins, weights=ii)
    counts, _ = np.histogram(rr, bins=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = sums / counts
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile[counts == 0] = np.nan
    valid = np.isfinite(profile)
    if np.any(valid):
        profile = np.interp(centers, centers[valid], profile[valid])
    else:
        profile[:] = 0.0
    return centers, profile


def create_2td_optical_system(
    design: dict,
    distances: Iterable[float],
    *,
    n_glass: float = DEFAULTS.n_glass,
    collimator_to_l1: float = DEFAULTS.collimator_to_l1,
):
    """
    Build a 2tD eigen_gbd setup from the F220APC output plane through targets.

    The F220APC-780 itself is represented by its collimated Gaussian output
    field at z=0. The downstream propagated optics are the two tabulated
    plano-aspheric shaper lenses.
    """
    _ensure_eigen_gbd_path()
    from eigen_gbd.optical_setups import OpticalSetup2tD
    from eigen_gbd.surfaces import Surface2tD
    from eigen_gbd.surface_types import Plane

    tab_surface = surface_type_2td()
    t1 = float(design["t1"])
    t2 = float(design["t2"])
    sep = float(design["separation"])
    ap1 = float(design["r"][-1])
    ap2 = float(design["R"][-1])
    z_l1_flat = float(collimator_to_l1)
    z_l1_asph = z_l1_flat + t1
    z_l2_asph = z_l1_flat + sep - t2
    z_l2_flat = z_l1_flat + sep

    surfaces = [
        Surface2tD(
            z0=0.0,
            surface_type=Plane,
            r_aperture=ap1 * 2.0,
            nIdx_before=1.0,
            nIdx_after=1.0,
            surf_plt_style_opts={"color": 0x8D99AE},
        ),
        Surface2tD(
            z0=z_l1_flat,
            surface_type=Plane,
            r_aperture=ap1,
            nIdx_before=1.0,
            nIdx_after=n_glass,
            surf_plt_style_opts={"color": 0x3A86FF},
        ),
        Surface2tD(
            z0=z_l1_asph,
            surface_type=tab_surface,
            surface_kwargs={"r_table": design["r"], "sag_table": design["sag1"], "sgn": -1},
            r_aperture=ap1,
            nIdx_before=n_glass,
            nIdx_after=1.0,
            surf_plt_style_opts={"color": 0x3A86FF},
        ),
        Surface2tD(
            z0=z_l2_asph,
            surface_type=tab_surface,
            surface_kwargs={"r_table": design["R"], "sag_table": design["sag2"], "sgn": -1},
            r_aperture=ap2,
            nIdx_before=1.0,
            nIdx_after=n_glass,
            surf_plt_style_opts={"color": 0x00C2A8},
        ),
        Surface2tD(
            z0=z_l2_flat,
            surface_type=Plane,
            r_aperture=ap2,
            nIdx_before=n_glass,
            nIdx_after=1.0,
            surf_plt_style_opts={"color": 0x00C2A8},
        ),
    ]

    target_indices = {}
    for distance in np.asarray(list(distances), dtype=float):
        surf_idx = len(surfaces)
        target_indices[float(distance)] = surf_idx
        surfaces.append(
            Surface2tD(
                z0=z_l2_flat + float(distance),
                surface_type=Plane,
                r_aperture=ap2 * 1.4,
                nIdx_before=1.0,
                nIdx_after=1.0,
                surf_plt_style_opts={"color": 0xFFD166},
            )
        )

    return OpticalSetup2tD(surfaces), target_indices


def run_2td_long_throw_simulation(
    design: dict,
    distances: Iterable[float] | None = None,
    *,
    n_glass: float = DEFAULTS.n_glass,
    n_gausslets: int = 529,
    n_polar_rings: int = 11,
    overlap_factor: float = 1.65,
    sampling_method: str = "polar_rings",
    enforce_radial_launch: bool = True,
    decomp_side: int = 45,
    eval_side: int = 55,
    radial_bins: int = 180,
) -> dict:
    """
    Propagate the F220APC-780 launch field through the actual 2tD lens surfaces.

    Distances are measured after the second lens flat face.
    """
    if distances is None:
        distances = np.array([0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00])
    distances = np.asarray(list(distances), dtype=float)

    _ensure_eigen_gbd_path()
    from eigen_gbd.cavities import build_gausslet2tD_field_matrix

    wavelength = float(design["wavelength"])
    launch_w0 = float(design["input_w0"])
    launch_radius = float(design["input_aperture_radius"])
    output_radius = float(design["output_radius"])
    eval_radius = output_radius * 1.15

    setup, target_indices = create_2td_optical_system(
        design,
        distances,
        n_glass=n_glass,
        collimator_to_l1=float(design.get("collimator_to_l1", DEFAULTS.collimator_to_l1)),
    )
    if sampling_method == "polar_rings":
        gausslets, launch_groups = _initialize_balanced_2td_gausslets(
            setup,
            wavelength=wavelength,
            launch_radius=launch_radius,
            overlap_factor=overlap_factor,
            n_rings=n_polar_rings,
        )
    else:
        gausslets = setup.initialize_Gausslets(
            surf_idx=0,
            n_gausslets=n_gausslets,
            overlap_factor=overlap_factor,
            wvln=wavelength,
            rng=launch_radius,
            sampling_method=sampling_method,
            initialize_on_curved_surface=False,
        )
        launch_groups = np.arange(len(gausslets), dtype=int)

    pts_decomp = _disc_points(launch_radius, decomp_side, 0.0)
    r_decomp = np.hypot(pts_decomp[:, 0], pts_decomp[:, 1])
    e_launch = np.exp(-(r_decomp / launch_w0) ** 2)
    for g in gausslets:
        g.compute_gouy_phases()
    g_launch = build_gausslet2tD_field_matrix(pts_decomp, gausslets, propLegIdx=0)
    if enforce_radial_launch and sampling_method == "polar_rings":
        ring_ids = np.unique(launch_groups)
        g_fit = np.column_stack([
            np.sum(g_launch[:, launch_groups == ring_id], axis=1)
            for ring_id in ring_ids
        ])
        coeffs_grouped, residuals, rank, singular_values = np.linalg.lstsq(g_fit, e_launch, rcond=1e-8)
        coeffs = np.empty(len(gausslets), dtype=complex)
        for ring_id, coeff in zip(ring_ids, coeffs_grouped):
            coeffs[launch_groups == ring_id] = coeff
        launch_fit = g_fit @ coeffs_grouped
    else:
        coeffs, residuals, rank, singular_values = np.linalg.lstsq(g_launch, e_launch, rcond=1e-8)
        launch_fit = g_launch @ coeffs
    for g, coeff in zip(gausslets, coeffs):
        g.ampl = coeff

    plane_pts0, x_grid, y_grid = _cartesian_plane_points(eval_radius, eval_side, 0.0)
    intensity_maps = []
    radial_profiles = []
    radial_axis = None
    segment_indices = []

    target_set = set(target_indices.values())
    for dest_idx in range(1, setup.num_surfaces):
        gausslets = setup.prop_Gausslets_to_surface(gausslets, dest_surf_idx=dest_idx)
        if dest_idx not in target_set:
            continue

        distance = next(d for d, idx in target_indices.items() if idx == dest_idx)
        z_target = setup.surfaces[dest_idx].z0
        segment_idx = len(gausslets[0].chief_ray_paths[0].ray_segments) - 2
        segment_indices.append(segment_idx)
        for g in gausslets:
            g.compute_gouy_phases()

        plane_pts = plane_pts0.copy()
        plane_pts[:, 2] = z_target
        g_eval = build_gausslet2tD_field_matrix(plane_pts, gausslets, propLegIdx=segment_idx)
        e_eval = g_eval @ np.ones(len(gausslets), dtype=complex)
        intensity = np.abs(e_eval.reshape(eval_side, eval_side)) ** 2
        peak = float(np.nanmax(intensity))
        if peak > 0:
            intensity = intensity / peak
        intensity_maps.append(intensity)

        r_prof, i_prof = _radial_average_map(
            x_grid,
            y_grid,
            intensity,
            r_max=eval_radius,
            n_bins=radial_bins,
        )
        radial_axis = r_prof
        radial_profiles.append(i_prof)

    intensity_maps = np.asarray(intensity_maps)
    radial_profiles = np.asarray(radial_profiles)
    if radial_axis is None:
        radial_axis = np.linspace(0.0, eval_radius, radial_bins)

    flat_radius = float(design["flat_radius"])
    edge_width = float(design["edge_width"])
    target_profile = soft_top_intensity(radial_axis, flat_radius, edge_width)

    rr_grid = np.hypot(x_grid, y_grid)
    core_mask = rr_grid <= 0.75 * flat_radius
    plateau_mask = rr_grid <= flat_radius
    cv_core = np.std(intensity_maps[:, core_mask], axis=1) / (
        np.mean(intensity_maps[:, core_mask], axis=1) + 1e-30
    )
    cv_plateau = np.std(intensity_maps[:, plateau_mask], axis=1) / (
        np.mean(intensity_maps[:, plateau_mask], axis=1) + 1e-30
    )
    rms_target = np.sqrt(np.mean((radial_profiles - target_profile[None, :]) ** 2, axis=1))

    launch_rmse = float(np.sqrt(np.mean(np.abs(launch_fit - e_launch) ** 2)))
    launch_rel_rmse = launch_rmse / (float(np.sqrt(np.mean(np.abs(e_launch) ** 2))) + 1e-30)

    return {
        "distances": distances,
        "x": x_grid[0],
        "y": y_grid[:, 0],
        "x_grid": x_grid,
        "y_grid": y_grid,
        "I_maps": intensity_maps,
        "r_profile": radial_axis,
        "I_profiles": radial_profiles,
        "I_target": target_profile,
        "cv_core": cv_core,
        "cv_plateau": cv_plateau,
        "rms_target": rms_target,
        "segment_indices": np.asarray(segment_indices),
        "launch_coeffs": coeffs,
        "launch_singular_values": singular_values,
        "launch_rank": np.asarray(rank),
        "launch_rel_rmse": np.asarray(launch_rel_rmse),
        "n_gausslets": np.asarray(len(gausslets)),
        "sampling_method": np.asarray(sampling_method),
        "enforce_radial_launch": np.asarray(enforce_radial_launch),
        "eval_radius": np.asarray(eval_radius),
    }


def _ideal_layout_paths(design: dict, n_rays: int = 25) -> list[np.ndarray]:
    """Generate idealized meridional ray paths for the system overview panel."""
    f_col = float(design["collimator_focal_length"])
    z_fiber = 0.0
    z_col = f_col
    z_l1_flat = z_col + float(design.get("collimator_to_l1", DEFAULTS.collimator_to_l1))
    z_l1_asph = z_l1_flat + float(design["t1"])
    z_l2_asph = z_l1_flat + float(design["separation"]) - float(design["t2"])
    z_l2_flat = z_l1_flat + float(design["separation"])
    z_end = z_l2_flat + 0.20

    ap1 = float(design["r"][-1])
    r_samples = np.linspace(-ap1, ap1, n_rays)
    paths = []
    for r_l1 in r_samples:
        if abs(r_l1) < 1e-15:
            r_l2 = 0.0
        else:
            r_l2 = np.sign(r_l1) * float(np.interp(abs(r_l1), design["r"], design["R"]))
        paths.append(np.array([
            [z_fiber, 0.0],
            [z_col, r_l1],
            [z_l1_flat, r_l1],
            [z_l1_asph - float(np.interp(abs(r_l1), design["r"], design["sag1"])), r_l1],
            [z_l2_asph - float(np.interp(abs(r_l2), design["R"], design["sag2"])), r_l2],
            [z_l2_flat, r_l2],
            [z_end, r_l2],
        ]))
    return paths


def make_long_throw_figure(design: dict, performance: dict) -> go.Figure:
    """Build the interactive Plotly performance visualizer."""
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
        ],
        row_heights=[0.40, 0.34, 0.26],
        vertical_spacing=0.09,
        horizontal_spacing=0.09,
        subplot_titles=[
            "PM780 collimator and two-lens shaper layout",
            "Radial intensity after L2",
            "Propagation heatmap",
            "Flatness metrics",
            "Design summary",
        ],
    )

    # Layout and rays.
    for path in _ideal_layout_paths(design):
        fig.add_trace(go.Scatter(
            x=path[:, 0] * 1e3,
            y=path[:, 1] * 1e3,
            mode="lines",
            line=dict(color="rgba(255,111,64,0.35)", width=1),
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=1)

    z_col = float(design["collimator_focal_length"]) * 1e3
    z_l1_flat = z_col + float(design.get("collimator_to_l1", DEFAULTS.collimator_to_l1)) * 1e3
    z_l1_asph = z_l1_flat + float(design["t1"]) * 1e3
    z_l2_asph = z_l1_flat + (float(design["separation"]) - float(design["t2"])) * 1e3
    z_l2_flat = z_l1_flat + float(design["separation"]) * 1e3
    first_plane = z_l2_flat + DEFAULTS.first_eval_distance * 1e3

    fig.add_trace(go.Scatter(
        x=[z_col, z_col],
        y=[-2.2, 2.2],
        mode="lines",
        line=dict(color="#6bb6ff", width=3),
        name="F220APC-780",
    ), row=1, col=1)

    r1 = design["r"] * 1e3
    sag1 = design["sag1"] * 1e3
    fig.add_trace(go.Scatter(
        x=z_l1_asph - sag1,
        y=r1,
        mode="lines",
        line=dict(color="#3a86ff", width=2),
        name="L1 asphere",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=z_l1_asph - sag1,
        y=-r1,
        mode="lines",
        line=dict(color="#3a86ff", width=2),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[z_l1_flat, z_l1_flat],
        y=[-r1[-1], r1[-1]],
        mode="lines",
        line=dict(color="#3a86ff", width=1),
        showlegend=False,
    ), row=1, col=1)

    r2 = design["R"] * 1e3
    sag2 = design["sag2"] * 1e3
    fig.add_trace(go.Scatter(
        x=z_l2_asph - sag2,
        y=r2,
        mode="lines",
        line=dict(color="#00c2a8", width=2),
        name="L2 asphere",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=z_l2_asph - sag2,
        y=-r2,
        mode="lines",
        line=dict(color="#00c2a8", width=2),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[z_l2_flat, z_l2_flat],
        y=[-r2[-1], r2[-1]],
        mode="lines",
        line=dict(color="#00c2a8", width=1),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[first_plane, first_plane],
        y=[-r2[-1] * 1.15, r2[-1] * 1.15],
        mode="lines",
        line=dict(color="#ffd166", width=2, dash="dash"),
        name="first spec plane",
    ), row=1, col=1)

    # Radial profiles.
    r_mm = performance["r_out"] * 1e3
    fig.add_trace(go.Scatter(
        x=r_mm,
        y=performance["I_target"],
        mode="lines",
        line=dict(color="black", width=2, dash="dot"),
        name="requested soft top",
    ), row=2, col=1)

    profile_indices = np.linspace(0, len(performance["distances"]) - 1, 6, dtype=int)
    colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
    for color, idx in zip(colors, profile_indices):
        distance = performance["distances"][idx]
        fig.add_trace(go.Scatter(
            x=r_mm,
            y=performance["I_profiles"][idx],
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{distance*1e3:.0f} mm",
        ), row=2, col=1)

    # Heatmap.
    fig.add_trace(go.Heatmap(
        x=r_mm,
        y=performance["distances"] * 1e3,
        z=performance["I_profiles"],
        colorscale="Viridis",
        colorbar=dict(title="I/Imax"),
        zmin=0,
        zmax=1,
        name="I(r,z)",
    ), row=2, col=2)

    # Metrics.
    fig.add_trace(go.Scatter(
        x=performance["distances"] * 1e3,
        y=performance["cv_core"] * 100,
        mode="lines+markers",
        line=dict(color="#2a9d8f", width=2.5),
        name="core CV",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=performance["distances"] * 1e3,
        y=performance["cv_plateau"] * 100,
        mode="lines+markers",
        line=dict(color="#e76f51", width=2.5),
        name="plateau CV",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=performance["distances"] * 1e3,
        y=performance["rms_target"] * 100,
        mode="lines+markers",
        line=dict(color="#6c757d", width=2.0, dash="dash"),
        name="RMS target error",
    ), row=3, col=1)

    # Summary table.
    first_idx = int(np.argmin(np.abs(performance["distances"] - DEFAULTS.first_eval_distance)))
    headers = ["Parameter", "Value"]
    values = [
        ["Input w0", f"{design['input_w0']*1e3:.3f} mm"],
        ["Flat radius", f"{design['flat_radius']*1e3:.2f} mm"],
        ["Edge width", f"{design['edge_width']*1e3:.2f} mm"],
        ["Output clear radius", f"{design['output_radius']*1e3:.2f} mm"],
        ["L1 diameter", f"{2*design['r'][-1]*1e3:.2f} mm"],
        ["L2 diameter", f"{2*design['R'][-1]*1e3:.2f} mm"],
        ["Lens separation", f"{design['separation']*1e3:.1f} mm"],
        ["OPL p-p", f"{np.ptp(design['opl'])/design['wavelength']:.3f} waves"],
        ["CV at 50 mm", f"{performance['cv_core'][first_idx]*100:.2f}% core"],
    ]
    fig.add_trace(go.Table(
        header=dict(values=headers, fill_color="#273043", font=dict(color="white")),
        cells=dict(values=list(map(list, zip(*values))), fill_color="#f7f8fa"),
    ), row=3, col=2)

    fig.update_xaxes(title_text="z from fiber (mm)", row=1, col=1)
    fig.update_yaxes(title_text="radius (mm)", row=1, col=1)
    fig.update_xaxes(title_text="radius (mm)", row=2, col=1)
    fig.update_yaxes(title_text="normalized intensity", range=[-0.04, 1.08], row=2, col=1)
    fig.update_xaxes(title_text="radius (mm)", row=2, col=2)
    fig.update_yaxes(title_text="distance after L2 (mm)", row=2, col=2)
    fig.update_xaxes(title_text="distance after L2 (mm)", row=3, col=1)
    fig.update_yaxes(title_text="metric (%)", row=3, col=1)

    fig.update_layout(
        title="PM780 long-throw soft flat-top shaper at 780 nm",
        height=1050,
        width=1200,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=85, b=45),
    )
    return fig


def make_2td_simulation_figure(design: dict, sim2td: dict) -> go.Figure:
    """Build an interactive visualizer from the 2tD GBD simulation results."""
    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
        ],
        row_heights=[0.28, 0.25, 0.25, 0.22],
        vertical_spacing=0.08,
        horizontal_spacing=0.09,
        subplot_titles=[
            "F220APC-780 launch and two-lens shaper layout",
            "2tD intensity at first plane",
            "2tD intensity at last plane",
            "Radial profiles from 2tD field",
            "Radial propagation heatmap",
            "Flatness metrics from 2tD field",
            "2tD run summary",
        ],
    )

    for path in _ideal_layout_paths(design):
        fig.add_trace(go.Scatter(
            x=path[:, 0] * 1e3,
            y=path[:, 1] * 1e3,
            mode="lines",
            line=dict(color="rgba(255,111,64,0.35)", width=1),
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=1)

    z_col = float(design["collimator_focal_length"]) * 1e3
    z_l1_flat = z_col + float(design.get("collimator_to_l1", DEFAULTS.collimator_to_l1)) * 1e3
    z_l1_asph = z_l1_flat + float(design["t1"]) * 1e3
    z_l2_asph = z_l1_flat + (float(design["separation"]) - float(design["t2"])) * 1e3
    z_l2_flat = z_l1_flat + float(design["separation"]) * 1e3
    first_plane = z_l2_flat + float(sim2td["distances"][0]) * 1e3

    fig.add_trace(go.Scatter(
        x=[z_col, z_col],
        y=[-2.2, 2.2],
        mode="lines",
        line=dict(color="#6bb6ff", width=3),
        name="F220APC-780",
    ), row=1, col=1)

    r1 = design["r"] * 1e3
    sag1 = design["sag1"] * 1e3
    fig.add_trace(go.Scatter(
        x=z_l1_asph - sag1,
        y=r1,
        mode="lines",
        line=dict(color="#3a86ff", width=2),
        name="L1",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=z_l1_asph - sag1,
        y=-r1,
        mode="lines",
        line=dict(color="#3a86ff", width=2),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[z_l1_flat, z_l1_flat],
        y=[-r1[-1], r1[-1]],
        mode="lines",
        line=dict(color="#3a86ff", width=1),
        showlegend=False,
    ), row=1, col=1)

    r2 = design["R"] * 1e3
    sag2 = design["sag2"] * 1e3
    fig.add_trace(go.Scatter(
        x=z_l2_asph - sag2,
        y=r2,
        mode="lines",
        line=dict(color="#00c2a8", width=2),
        name="L2",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=z_l2_asph - sag2,
        y=-r2,
        mode="lines",
        line=dict(color="#00c2a8", width=2),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[z_l2_flat, z_l2_flat],
        y=[-r2[-1], r2[-1]],
        mode="lines",
        line=dict(color="#00c2a8", width=1),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[first_plane, first_plane],
        y=[-r2[-1] * 1.15, r2[-1] * 1.15],
        mode="lines",
        line=dict(color="#ffd166", width=2, dash="dash"),
        name="first 2tD plane",
    ), row=1, col=1)

    x_mm = sim2td["x"] * 1e3
    y_mm = sim2td["y"] * 1e3
    first_idx = 0
    last_idx = len(sim2td["distances"]) - 1
    for idx, cell in [(first_idx, (2, 1)), (last_idx, (2, 2))]:
        fig.add_trace(go.Heatmap(
            x=x_mm,
            y=y_mm,
            z=sim2td["I_maps"][idx],
            zmin=0,
            zmax=1,
            colorscale="Magma",
            colorbar=dict(title="I/Imax", len=0.25),
            name=f"{sim2td['distances'][idx]*1e3:.0f} mm",
        ), row=cell[0], col=cell[1])

    r_mm = sim2td["r_profile"] * 1e3
    fig.add_trace(go.Scatter(
        x=r_mm,
        y=sim2td["I_target"],
        mode="lines",
        line=dict(color="black", width=2, dash="dot"),
        name="requested soft top",
    ), row=3, col=1)
    profile_indices = np.unique(np.linspace(0, len(sim2td["distances"]) - 1, 6, dtype=int))
    colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
    for color, idx in zip(colors, profile_indices):
        fig.add_trace(go.Scatter(
            x=r_mm,
            y=sim2td["I_profiles"][idx],
            mode="lines",
            line=dict(color=color, width=2),
            name=f"2tD {sim2td['distances'][idx]*1e3:.0f} mm",
        ), row=3, col=1)

    fig.add_trace(go.Heatmap(
        x=r_mm,
        y=sim2td["distances"] * 1e3,
        z=sim2td["I_profiles"],
        zmin=0,
        zmax=1,
        colorscale="Viridis",
        colorbar=dict(title="I/Imax", len=0.25),
        name="I(r,z)",
    ), row=3, col=2)

    fig.add_trace(go.Scatter(
        x=sim2td["distances"] * 1e3,
        y=sim2td["cv_core"] * 100,
        mode="lines+markers",
        line=dict(color="#2a9d8f", width=2.5),
        name="2tD core CV",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=sim2td["distances"] * 1e3,
        y=sim2td["cv_plateau"] * 100,
        mode="lines+markers",
        line=dict(color="#e76f51", width=2.5),
        name="2tD plateau CV",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=sim2td["distances"] * 1e3,
        y=sim2td["rms_target"] * 100,
        mode="lines+markers",
        line=dict(color="#6c757d", width=2.0, dash="dash"),
        name="2tD RMS target error",
    ), row=4, col=1)

    first_spec_idx = int(np.argmin(np.abs(sim2td["distances"] - DEFAULTS.first_eval_distance)))
    best_idx = int(np.argmin(sim2td["cv_core"]))
    values = [
        ["Collimator", str(design["collimator_model"])],
        ["f", f"{design['collimator_focal_length']*1e3:.2f} mm"],
        ["NA", f"{design['collimator_na']:.2f}"],
        ["Launch w0", f"{design['input_w0']*1e3:.3f} mm"],
        ["Gausslets", f"{int(np.asarray(sim2td['n_gausslets']))}"],
        ["Sampling", str(np.asarray(sim2td.get("sampling_method", "unknown")))],
        ["Radial launch", str(bool(np.asarray(sim2td.get("enforce_radial_launch", False))))],
        ["Launch fit RMSE", f"{float(np.asarray(sim2td['launch_rel_rmse']))*100:.2f}%"],
        ["Core CV at 50 mm", f"{sim2td['cv_core'][first_spec_idx]*100:.2f}%"],
        ["Plateau CV at 50 mm", f"{sim2td['cv_plateau'][first_spec_idx]*100:.2f}%"],
        ["Best core CV", f"{sim2td['cv_core'][best_idx]*100:.2f}% at {sim2td['distances'][best_idx]*1e3:.0f} mm"],
    ]
    fig.add_trace(go.Table(
        header=dict(values=["Parameter", "Value"], fill_color="#273043", font=dict(color="white")),
        cells=dict(values=list(map(list, zip(*values))), fill_color="#f7f8fa"),
    ), row=4, col=2)

    fig.update_xaxes(title_text="z from fiber (mm)", row=1, col=1)
    fig.update_yaxes(title_text="radius (mm)", row=1, col=1)
    fig.update_xaxes(title_text="x (mm)", row=2, col=1)
    fig.update_yaxes(title_text="y (mm)", scaleanchor="x2", scaleratio=1, row=2, col=1)
    fig.update_xaxes(title_text="x (mm)", row=2, col=2)
    fig.update_yaxes(title_text="y (mm)", scaleanchor="x3", scaleratio=1, row=2, col=2)
    fig.update_xaxes(title_text="radius (mm)", row=3, col=1)
    fig.update_yaxes(title_text="normalized intensity", range=[-0.04, 1.08], row=3, col=1)
    fig.update_xaxes(title_text="radius (mm)", row=3, col=2)
    fig.update_yaxes(title_text="distance after L2 (mm)", row=3, col=2)
    fig.update_xaxes(title_text="distance after L2 (mm)", row=4, col=1)
    fig.update_yaxes(title_text="metric (%)", row=4, col=1)
    fig.update_layout(
        title="2tD GBD performance: PM780/F220APC-780 long-throw flat-top shaper",
        height=1320,
        width=1220,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=95, b=45),
    )
    return fig


def write_2td_summary_png(design: dict, sim2td: dict, output_path: str | Path) -> Path:
    """Write a compact static PNG summary of the 2tD performance."""
    output_path = Path(output_path)
    first_idx = 0
    last_idx = len(sim2td["distances"]) - 1
    r_mm = sim2td["r_profile"] * 1e3
    x_mm = sim2td["x"] * 1e3
    y_mm = sim2td["y"] * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    im0 = axes[0, 0].imshow(
        sim2td["I_maps"][first_idx],
        extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
        origin="lower",
        vmin=0,
        vmax=1,
        cmap="magma",
    )
    axes[0, 0].set_title(f"2tD intensity, {sim2td['distances'][first_idx]*1e3:.0f} mm after L2")
    axes[0, 0].set_xlabel("x (mm)")
    axes[0, 0].set_ylabel("y (mm)")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0, 0], label="I/Imax")

    im1 = axes[0, 1].imshow(
        sim2td["I_maps"][last_idx],
        extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
        origin="lower",
        vmin=0,
        vmax=1,
        cmap="magma",
    )
    axes[0, 1].set_title(f"2tD intensity, {sim2td['distances'][last_idx]*1e3:.0f} mm after L2")
    axes[0, 1].set_xlabel("x (mm)")
    axes[0, 1].set_ylabel("y (mm)")
    axes[0, 1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[0, 1], label="I/Imax")

    axes[1, 0].plot(r_mm, sim2td["I_target"], "k:", lw=2, label="requested")
    for idx in np.unique(np.linspace(0, len(sim2td["distances"]) - 1, 5, dtype=int)):
        axes[1, 0].plot(r_mm, sim2td["I_profiles"][idx], lw=1.8, label=f"{sim2td['distances'][idx]*1e3:.0f} mm")
    axes[1, 0].set_title("Azimuthal radial profiles")
    axes[1, 0].set_xlabel("radius (mm)")
    axes[1, 0].set_ylabel("I/Imax")
    axes[1, 0].set_ylim(-0.04, 1.08)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(sim2td["distances"] * 1e3, sim2td["cv_core"] * 100, "o-", label="core CV")
    axes[1, 1].plot(sim2td["distances"] * 1e3, sim2td["cv_plateau"] * 100, "o-", label="plateau CV")
    axes[1, 1].plot(sim2td["distances"] * 1e3, sim2td["rms_target"] * 100, "o--", label="RMS target error")
    axes[1, 1].set_title("2tD flatness metrics")
    axes[1, 1].set_xlabel("distance after L2 (mm)")
    axes[1, 1].set_ylabel("metric (%)")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"{design['collimator_model']} launch, {design['wavelength']*1e9:.0f} nm")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_2td_1d_profile_figure(design: dict, sim2td: dict) -> go.Figure:
    """Plot 1D radial intensity profiles from the 2tD GBD maps."""
    fig = go.Figure()
    r_mm = sim2td["r_profile"] * 1e3
    fig.add_trace(go.Scatter(
        x=r_mm,
        y=sim2td["I_target"],
        mode="lines",
        line=dict(color="black", width=2.5, dash="dot"),
        name="requested soft top",
    ))

    distances = sim2td["distances"]
    colors = ["#264653", "#287271", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51", "#b56576", "#6d597a"]
    for idx, distance in enumerate(distances):
        fig.add_trace(go.Scatter(
            x=r_mm,
            y=sim2td["I_profiles"][idx],
            mode="lines",
            line=dict(color=colors[idx % len(colors)], width=2),
            name=f"{distance*1e3:.0f} mm after L2",
        ))

    flat_mm = float(design["flat_radius"]) * 1e3
    fig.add_vline(x=flat_mm, line_width=1.5, line_dash="dash", line_color="#6c757d")
    fig.update_layout(
        title="1D radial intensity profiles from 2tD GBD propagation",
        template="plotly_white",
        width=1050,
        height=650,
        xaxis_title="radius (mm)",
        yaxis_title="normalized intensity",
        yaxis=dict(range=[-0.04, 1.08]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=65, r=30, t=90, b=55),
    )
    return fig


def write_2td_1d_profiles_csv(sim2td: dict, output_path: str | Path) -> Path:
    """Write the 2tD radial profile data used in the 1D profile plot."""
    output_path = Path(output_path)
    header = ["r_mm", "requested"]
    columns = [sim2td["r_profile"] * 1e3, sim2td["I_target"]]
    for distance, profile in zip(sim2td["distances"], sim2td["I_profiles"]):
        header.append(f"I_{distance*1e3:.0f}mm_after_L2")
        columns.append(profile)
    data = np.column_stack(columns)
    np.savetxt(output_path, data, delimiter=",", header=",".join(header), comments="")
    return output_path


def write_long_throw_outputs(
    output_dir: str | Path = "output_long_throw_pm780",
    *,
    design: dict | None = None,
    performance: dict | None = None,
    sim2td: dict | None = None,
    run_2td: bool = True,
) -> dict:
    """Generate design files, STL meshes, visualizers, and a short spec file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if design is None:
        design = design_pm780_long_throw_shaper()
    if performance is None:
        metric_distances = np.linspace(DEFAULTS.first_eval_distance, 1.0, 25)
        performance = simulate_long_throw_performance(design, distances=metric_distances)

    fig = make_long_throw_figure(design, performance)
    html_path = out / "pm780_long_throw_visualizer.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(out / "pm780_long_throw_visualizer.png", scale=2)
    except Exception:
        pass

    write_sag_csv(out / "lens1_sag.csv", design["r"], design["sag1"], "PM780 long-throw L1")
    write_sag_csv(out / "lens2_sag.csv", design["R"], design["sag2"], "PM780 long-throw L2")
    stl_paths = export_lens_stls(design, out)

    sim2td_html_path = None
    sim2td_png_path = None
    sim2td_profiles_html_path = None
    sim2td_profiles_png_path = None
    sim2td_profiles_csv_path = None
    if run_2td:
        if sim2td is None:
            sim2td = run_2td_long_throw_simulation(design)
        sim2td_fig = make_2td_simulation_figure(design, sim2td)
        sim2td_html_path = out / "pm780_long_throw_2td_gbd.html"
        sim2td_fig.write_html(sim2td_html_path, include_plotlyjs="cdn")
        sim2td_png_path = write_2td_summary_png(design, sim2td, out / "pm780_long_throw_2td_gbd_summary.png")
        try:
            sim2td_fig.write_image(out / "pm780_long_throw_2td_gbd.png", scale=2)
        except Exception:
            pass
        sim2td_profiles_fig = make_2td_1d_profile_figure(design, sim2td)
        sim2td_profiles_html_path = out / "pm780_long_throw_2td_1d_profiles.html"
        sim2td_profiles_png_path = out / "pm780_long_throw_2td_1d_profiles.png"
        sim2td_profiles_csv_path = write_2td_1d_profiles_csv(
            sim2td,
            out / "pm780_long_throw_2td_1d_profiles.csv",
        )
        sim2td_profiles_fig.write_html(sim2td_profiles_html_path, include_plotlyjs="cdn")
        try:
            sim2td_profiles_fig.write_image(sim2td_profiles_png_path, scale=2)
        except Exception:
            sim2td_profiles_png_path = None

    np.savez_compressed(
        out / "performance_data.npz",
        **{k: v for k, v in performance.items() if isinstance(v, np.ndarray)},
        r=design["r"],
        R=design["R"],
        sag1=design["sag1"],
        sag2=design["sag2"],
        opl=design["opl"],
    )
    if sim2td is not None:
        np.savez_compressed(
            out / "performance_2td_gbd_data.npz",
            **{k: v for k, v in sim2td.items() if isinstance(v, np.ndarray)},
        )

    first_idx = int(np.argmin(np.abs(performance["distances"] - DEFAULTS.first_eval_distance)))
    if sim2td is not None:
        first_2td_idx = int(np.argmin(np.abs(sim2td["distances"] - DEFAULTS.first_eval_distance)))
        sim2td_block = f"""
## 2tD GBD Propagation Check

- Simulation basis: {int(np.asarray(sim2td['n_gausslets']))} 2tD gausslets on a {str(np.asarray(sim2td.get('sampling_method', 'unknown')))} launch grid, fitted to the F220APC-780 collimated Gaussian launch.
- Radial launch coefficients enforced: {bool(np.asarray(sim2td.get('enforce_radial_launch', False)))}.
- Launch basis relative RMSE: {float(np.asarray(sim2td['launch_rel_rmse']))*100:.2f}%.
- Core CV at 50 mm: {sim2td['cv_core'][first_2td_idx]*100:.2f}%.
- Full nominal plateau CV at 50 mm: {sim2td['cv_plateau'][first_2td_idx]*100:.2f}%.
- 2tD visualizer: `{sim2td_html_path.name if sim2td_html_path else 'not generated'}`.
- 2tD static plot: `{sim2td_png_path.name if sim2td_png_path else 'not generated'}`.
- 1D radial profile plot: `{sim2td_profiles_html_path.name if sim2td_profiles_html_path else 'not generated'}`.
"""
    else:
        sim2td_block = """
## 2tD GBD Propagation Check

- Not generated in this run.
"""
    spec = f"""# PM780 Long-Throw Soft Flat-Top Shaper

## Optical Intent

- Source: PM780 fiber, MFD {design['fiber_mfd']*1e6:.2f} um.
- Collimator: {design['collimator_model']}, f = {design['collimator_focal_length']*1e3:.2f} mm, NA = {design['collimator_na']:.2f}.
- Collimated Gaussian radius: w0 = {design['input_w0']*1e3:.3f} mm.
- Distance from collimator output plane to L1 flat face: {design['collimator_to_l1']*1e3:.1f} mm.
- Output profile: Fermi-Dirac soft flat-top.
- Nominal flat radius: {design['flat_radius']*1e3:.2f} mm.
- Edge rolloff width: {design['edge_width']*1e3:.2f} mm.
- First required flat-top plane: {DEFAULTS.first_eval_distance*1e3:.0f} mm after L2.

## Lens Pair

- L1 diameter: {2*design['r'][-1]*1e3:.2f} mm.
- L2 diameter: {2*design['R'][-1]*1e3:.2f} mm.
- Center thicknesses: L1 {design['t1']*1e3:.2f} mm, L2 {design['t2']*1e3:.2f} mm.
- Lens-center separation: {design['separation']*1e3:.1f} mm.
- Assumed refractive index: {DEFAULTS.n_glass:.4f}.
- OPL peak-to-peak: {np.ptp(design['opl'])/design['wavelength']:.3f} waves.
- L1 STL: `{stl_paths['lens1_stl'].name}`.
- L2 STL: `{stl_paths['lens2_stl'].name}`.
- STL coordinate units: millimeters.

## Scalar Diffraction Proxy

- Core CV at 50 mm: {performance['cv_core'][first_idx]*100:.2f}%.
- Full nominal plateau CV at 50 mm: {performance['cv_plateau'][first_idx]*100:.2f}%.
- Scalar visualizer: `{html_path.name}`.
{sim2td_block}
## Notes

The scalar check models the intended second-lens output field with flat phase.
The 2tD GBD check propagates the fitted F220APC-780 Gaussian launch through the
actual tabulated lens surfaces in eigen_gbd.
The F220APC-780 package is represented as the post-collimator Gaussian launch
plane using the catalog focal length and NA; its internal prescription is not
traced in this design pass.
"""
    spec_path = out / "design_spec.md"
    spec_path.write_text(spec)

    return {
        "design": design,
        "performance": performance,
        "sim2td": sim2td,
        "html_path": html_path,
        "sim2td_html_path": sim2td_html_path,
        "sim2td_png_path": sim2td_png_path,
        "sim2td_profiles_html_path": sim2td_profiles_html_path,
        "sim2td_profiles_png_path": sim2td_profiles_png_path,
        "sim2td_profiles_csv_path": sim2td_profiles_csv_path,
        **stl_paths,
        "spec_path": spec_path,
    }


def main() -> None:
    write_long_throw_outputs()


if __name__ == "__main__":
    main()
