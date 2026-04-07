"""
Ray tracer using tabulated (point-cloud) sag profiles.

Instead of fitting surfaces to even asphere polynomials, this module
interpolates directly from the sag and dsag/dr tables computed by the
analytical ODE solver. This eliminates polynomial fit error entirely,
giving better collimation and uniformity.

Suitable for fabrication on:
- Two-photon polymerization (2PP) printers
- Diamond turning lathes
- Single-point diamond milling

Both can accept arbitrary surface profiles as point clouds or splines.
"""

import jax
import jax.numpy as jnp
from functools import partial


def _interp_sag(r, r_table, sag_table):
    """Interpolate sag from a tabulated profile. Piecewise linear."""
    return jnp.interp(r, r_table, sag_table)


def _interp_deriv(r, r_table, dsag_table):
    """Interpolate dsag/dr from a tabulated derivative profile."""
    return jnp.interp(r, r_table, dsag_table)


def _ray_table_intersect(origin, direction, r_table, sag_table, dsag_table, z_vertex, n_iter=30):
    """
    Find intersection of a ray with a tabulated aspheric surface.

    Surface: z = z_vertex - sag(r), where sag and dsag/dr are
    interpolated from precomputed tables.
    Uses Newton's method with fixed iteration count.
    """
    t = (z_vertex - origin[2]) / (direction[2] + 1e-30)

    def newton_step(t, _):
        p = origin + t * direction
        r = jnp.sqrt(p[0] ** 2 + p[1] ** 2)
        s = _interp_sag(r, r_table, sag_table)
        f = p[2] - z_vertex + s

        ds_dr = _interp_deriv(r, r_table, dsag_table)
        dr_dt = (p[0] * direction[0] + p[1] * direction[1]) / (r + 1e-30)
        df_dt = direction[2] + ds_dr * dr_dt
        t = t - f / (df_dt + 1e-30)
        return t, None

    t, _ = jax.lax.scan(newton_step, t, None, length=n_iter)
    hit = origin + t * direction
    return t, hit


def _surface_normal_table(point, r_table, dsag_table, z_vertex):
    """
    Surface normal from tabulated dsag/dr.

    Surface: z = z_vertex - sag(r), so F = z - z_vertex + sag(r) = 0.
    Gradient: (dsag/dr * cos_t, dsag/dr * sin_t, 1) = direction of +Z.
    """
    x, y = point[0], point[1]
    r = jnp.sqrt(x ** 2 + y ** 2)
    dz_dr = _interp_deriv(r, r_table, dsag_table)

    cos_t = x / (r + 1e-30)
    sin_t = y / (r + 1e-30)

    nx = dz_dr * cos_t
    ny = dz_dr * sin_t
    nz = 1.0

    norm = jnp.sqrt(nx * nx + ny * ny + nz * nz)
    return jnp.array([nx / norm, ny / norm, nz / norm])


def _snell_refract(direction, normal, n1, n2):
    """Snell's law refraction. Normal points toward SOURCE medium."""
    d = direction / (jnp.linalg.norm(direction) + 1e-30)
    n = normal / (jnp.linalg.norm(normal) + 1e-30)
    cosi = -jnp.dot(d, n)
    cosi = jnp.clip(cosi, -1.0, 1.0)
    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    weight = jax.nn.sigmoid(k * 1e4)
    k_safe = jnp.maximum(k, 1e-8)
    t = eta * d + (eta * cosi - jnp.sqrt(k_safe)) * n
    t = t / (jnp.linalg.norm(t) + 1e-30)
    return t, weight


def make_two_lens_tracer(design_result, n_glass=1.52):
    """
    Create a JIT-compiled ray tracer for a two-lens beam shaper
    using tabulated sag profiles directly from the ODE solution.

    Args:
        design_result: dict from design_two_lens_shaper()
        n_glass: refractive index

    Returns:
        trace_fn(origins, directions) -> (exit_positions, exit_directions)
    """
    t1 = design_result['t1']
    t2 = design_result['t2']
    sep = design_result['separation']

    # Surface 1: parameterized by input radius r
    r1_table = jnp.array(design_result['r'])
    sag1_table = jnp.array(design_result['sag1'])
    dsag1_table = jnp.array(design_result.get('dsag1', jnp.gradient(sag1_table, r1_table)))

    # Surface 2: parameterized by output radius R
    r2_table = jnp.array(design_result['R'])
    sag2_table = jnp.array(design_result['sag2'])
    dsag2_table = jnp.array(design_result.get('dsag2', jnp.gradient(sag2_table, r2_table)))

    z1_asph = t1        # lens 1 aspheric vertex
    z2_asph = sep - t2  # lens 2 aspheric vertex

    def trace_single(origin, direction):
        # 1. Flat face of lens 1 at z=0 (air -> glass)
        t_hit = -origin[2] / (direction[2] + 1e-30)
        hit1 = origin + t_hit * direction
        d_g1, _ = _snell_refract(direction, jnp.array([0., 0., -1.]), 1.0, n_glass)

        # 2. Aspheric face of lens 1 (glass -> air)
        _, hit2 = _ray_table_intersect(hit1, d_g1, r1_table, sag1_table, dsag1_table, z1_asph)
        n1 = _surface_normal_table(hit2, r1_table, dsag1_table, z1_asph)
        d_air, _ = _snell_refract(d_g1, -n1, n_glass, 1.0)

        # 3. Aspheric face of lens 2 (air -> glass)
        _, hit3 = _ray_table_intersect(hit2, d_air, r2_table, sag2_table, dsag2_table, z2_asph)
        n2 = _surface_normal_table(hit3, r2_table, dsag2_table, z2_asph)
        d_g2, _ = _snell_refract(d_air, -n2, 1.0, n_glass)

        # 4. Flat face of lens 2 at z=sep (glass -> air)
        t4 = (sep - hit3[2]) / (d_g2[2] + 1e-30)
        hit4 = hit3 + t4 * d_g2
        d_out, _ = _snell_refract(d_g2, jnp.array([0., 0., -1.]), n_glass, 1.0)

        r_in = jnp.sqrt(hit1[0] ** 2 + hit1[1] ** 2)
        return hit4, d_out, r_in

    @jax.jit
    def trace_bundle(origins, directions):
        return jax.vmap(trace_single)(origins, directions)

    return trace_bundle


def make_single_lens_tracer(design_result, center_thickness, n_glass=1.52, target_z=0.25):
    """
    Create a JIT-compiled ray tracer for a single-lens beam shaper
    using the tabulated sag profile.

    Args:
        design_result: dict from analytical.design_beam_shaper()
        center_thickness: lens center thickness
        n_glass: refractive index
        target_z: target observation plane

    Returns:
        trace_fn(origins, directions) -> (target_positions, weights, input_radii)
    """
    r_table = jnp.array(design_result['r'])
    sag_table = jnp.array(design_result['sag'])
    dsag_table = jnp.gradient(sag_table, r_table)

    z_flat = -center_thickness / 2.0
    z_asph = center_thickness / 2.0
    ap = float(r_table[-1])

    def trace_single(origin, direction):
        # 1. Flat face (air -> glass)
        t1 = (z_flat - origin[2]) / (direction[2] + 1e-30)
        hit1 = origin + t1 * direction
        r_hit = jnp.sqrt(hit1[0] ** 2 + hit1[1] ** 2)
        aperture_weight = jax.nn.sigmoid((ap - r_hit) * 1e3)

        flat_n = jnp.array([0.0, 0.0, -1.0])
        d_glass, w1 = _snell_refract(direction, flat_n, 1.0, n_glass)

        # 2. Aspheric face (glass -> air) — tabulated
        _, hit2 = _ray_table_intersect(hit1, d_glass, r_table, sag_table, dsag_table, z_asph)
        n_out = _surface_normal_table(hit2, r_table, dsag_table, z_asph)
        d_air, w2 = _snell_refract(d_glass, -n_out, n_glass, 1.0)

        # 3. Propagate to target
        t3 = (target_z - hit2[2]) / (d_air[2] + 1e-30)
        target = hit2 + t3 * d_air

        weight = aperture_weight * w1 * w2
        input_r = jnp.sqrt(hit1[0] ** 2 + hit1[1] ** 2)
        return target[:2], weight, input_r

    @jax.jit
    def trace_bundle(origins, directions):
        return jax.vmap(trace_single)(origins, directions)

    return trace_bundle


def extract_paths_two_lens(design_result, origins, directions, n_glass=1.52, target_z_beyond=0.0):
    """Extract ray paths through a two-lens system for visualization."""
    trace_fn = make_two_lens_tracer(design_result, n_glass)
    sep = design_result['separation']

    t1 = design_result['t1']
    t2 = design_result['t2']
    r1_table = jnp.array(design_result['r'])
    sag1_table = jnp.array(design_result['sag1'])
    r2_table = jnp.array(design_result['R'])
    sag2_table = jnp.array(design_result['sag2'])
    dsag1_table = jnp.gradient(sag1_table, r1_table)
    dsag2_table = jnp.gradient(sag2_table, r2_table)

    def trace_path(origin, direction):
        t_hit = -origin[2] / (direction[2] + 1e-30)
        hit1 = origin + t_hit * direction
        d_g1, _ = _snell_refract(direction, jnp.array([0., 0., -1.]), 1.0, n_glass)
        _, hit2 = _ray_table_intersect(hit1, d_g1, r1_table, sag1_table, dsag1_table, t1)
        n1 = _surface_normal_table(hit2, r1_table, dsag1_table, t1)
        d_air, _ = _snell_refract(d_g1, -n1, n_glass, 1.0)
        _, hit3 = _ray_table_intersect(hit2, d_air, r2_table, sag2_table, dsag2_table, sep - t2)
        n2 = _surface_normal_table(hit3, r2_table, dsag2_table, sep - t2)
        d_g2, _ = _snell_refract(d_air, -n2, 1.0, n_glass)
        t4 = (sep - hit3[2]) / (d_g2[2] + 1e-30)
        hit4 = hit3 + t4 * d_g2
        d_out, _ = _snell_refract(d_g2, jnp.array([0., 0., -1.]), n_glass, 1.0)
        # Propagate to target_z_beyond past lens 2
        t5 = target_z_beyond / (d_out[2] + 1e-30)
        hit5 = hit4 + t5 * d_out
        return jnp.stack([origin, hit1, hit2, hit3, hit4, hit5])

    return jax.vmap(trace_path)(origins, directions)
