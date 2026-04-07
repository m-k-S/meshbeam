"""
Aspheric surface definition and triangle mesh generation.

Standard even asphere sag equation:
    z(r) = c*r^2 / (1 + sqrt(1 - (1+k)*c^2*r^2)) + sum(alpha_i * r^(2i+4))

where c = 1/R (curvature), k = conic constant, alpha_i are polynomial coefficients.
The polynomial term starts at r^4 (i=0 -> r^4, i=1 -> r^6, etc.)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AsphericProfile:
    """An even aspheric surface profile."""
    curvature: float              # c = 1/R  (positive = convex toward +Z)
    conic_constant: float = 0.0   # k (0=sphere, -1=paraboloid)
    alphas: Tuple[float, ...] = ()  # higher-order even polynomial coefficients (r^4, r^6, ...)
    aperture_radius: float = 50e-3


def aspheric_sag(r: np.ndarray, profile: AsphericProfile) -> np.ndarray:
    """Evaluate aspheric sag z(r) at radial distance(s) r."""
    r = np.asarray(r, dtype=float)
    c = profile.curvature
    k = profile.conic_constant

    # Conic term: c*r^2 / (1 + sqrt(1 - (1+k)*c^2*r^2))
    r2 = r * r
    denom_arg = 1.0 - (1.0 + k) * c * c * r2
    # Clamp to avoid sqrt of negative (outside valid aperture)
    denom_arg = np.maximum(denom_arg, 0.0)
    z = c * r2 / (1.0 + np.sqrt(denom_arg))

    # Polynomial correction terms: alpha_0 * r^4 + alpha_1 * r^6 + ...
    r_power = r2 * r2  # r^4
    for alpha in profile.alphas:
        z = z + alpha * r_power
        r_power = r_power * r2  # next even power

    return z


def aspheric_deriv(r: np.ndarray, profile: AsphericProfile) -> np.ndarray:
    """Analytic derivative dz/dr of the aspheric sag."""
    r = np.asarray(r, dtype=float)
    c = profile.curvature
    k = profile.conic_constant

    r2 = r * r
    denom_arg = 1.0 - (1.0 + k) * c * c * r2
    denom_arg = np.maximum(denom_arg, 1e-30)
    sqrt_denom = np.sqrt(denom_arg)

    # Derivative of conic term: c*r / sqrt(1 - (1+k)*c^2*r^2)
    dz_dr = c * r / sqrt_denom

    # Polynomial terms: d/dr(alpha_i * r^(2i+4)) = alpha_i * (2i+4) * r^(2i+3)
    r_power = r2 * r  # r^3
    for i, alpha in enumerate(profile.alphas):
        exponent = 2 * i + 4
        dz_dr = dz_dr + alpha * exponent * r_power
        r_power = r_power * r2  # next odd power

    return dz_dr


def aspheric_normal(r: np.ndarray, theta: np.ndarray, profile: AsphericProfile) -> np.ndarray:
    """
    Compute outward surface normal at points on the aspheric surface.

    The surface is z = z_vertex - sag(r) (drops below vertex for convex).
    Returns shape (N, 3) array of unit normals pointing away from the lens
    (in the +Z direction at the vertex).
    """
    dz_dr = aspheric_deriv(r, profile)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Surface: F = z - z_vertex + sag(r) = 0
    # Gradient: (dsag/dr * x/r, dsag/dr * y/r, 1) = outward normal
    nx = dz_dr * cos_t
    ny = dz_dr * sin_t
    nz = np.ones_like(r)

    # Normalize
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / norm, ny / norm, nz / norm], axis=-1)


def plano_aspheric_tris(
    profile: AsphericProfile,
    center_thickness: float,
    radial_segments: int = 48,
    azimuth_segments: int = 180,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Generate triangulated plano-aspheric lens mesh.

    The aspheric face is at +Z (toward the incoming beam), flat face at -Z.
    Follows the same polar grid and triangle winding conventions as
    mesh-optics-master/geometry.py:plano_convex_tris.

    Returns array of triangles with shape (N, 3, 3).
    """
    if radial_segments < 2 or azimuth_segments < 8:
        raise ValueError("radial_segments >= 2 and azimuth_segments >= 8 required.")

    cx, cy, cz = center
    a = profile.aperture_radius
    if a <= 0:
        raise ValueError("aperture_radius must be > 0.")

    z_plane = cz - center_thickness * 0.5   # flat face
    z_vertex = cz + center_thickness * 0.5   # aspheric vertex

    def z_asph(r: np.ndarray) -> np.ndarray:
        return z_vertex - aspheric_sag(r, profile)

    # Validate: edge of aspheric surface must be above the flat face
    edge_z = float(z_asph(np.array([a]))[0])
    if edge_z <= z_plane:
        raise ValueError(
            f"Invalid geometry: aspheric edge z={edge_z:.6g} <= plane z={z_plane:.6g}. "
            "Decrease aperture, increase |R|, or increase center_thickness."
        )

    # Build polar grids
    rs = np.linspace(0.0, a, radial_segments + 1)
    thetas = np.linspace(0.0, 2.0 * np.pi, azimuth_segments, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    def ring_xy(r):
        return (cx + r * cos_t, cy + r * sin_t)

    # Aspheric surface points
    asph_pts = []
    for r in rs:
        x, y = ring_xy(r)
        z = np.full_like(x, float(z_asph(np.array([r]))[0]))
        asph_pts.append(np.stack([x, y, z], axis=-1))
    asph_pts = np.stack(asph_pts, axis=0)

    # Planar surface points
    plane_pts = []
    for r in rs:
        x, y = ring_xy(r)
        z = np.full_like(x, z_plane)
        plane_pts.append(np.stack([x, y, z], axis=-1))
    plane_pts = np.stack(plane_pts, axis=0)

    tris = []

    def add_quad(v00, v10, v11, v01, outward_ccw=True):
        if outward_ccw:
            tris.append(np.array([v00, v10, v11], dtype=float))
            tris.append(np.array([v00, v11, v01], dtype=float))
        else:
            tris.append(np.array([v00, v11, v10], dtype=float))
            tris.append(np.array([v00, v01, v11], dtype=float))

    # Aspheric cap (outward normals point +Z)
    for j in range(radial_segments):
        for k in range(azimuth_segments):
            k2 = (k + 1) % azimuth_segments
            v00 = asph_pts[j, k]
            v10 = asph_pts[j + 1, k]
            v11 = asph_pts[j + 1, k2]
            v01 = asph_pts[j, k2]
            add_quad(v00, v10, v11, v01, outward_ccw=True)

    # Plane (outward = -Z)
    for j in range(radial_segments):
        for k in range(azimuth_segments):
            k2 = (k + 1) % azimuth_segments
            v00 = plane_pts[j, k]
            v10 = plane_pts[j + 1, k]
            v11 = plane_pts[j + 1, k2]
            v01 = plane_pts[j, k2]
            add_quad(v00, v10, v11, v01, outward_ccw=False)

    # Cylindrical rim (outward radial)
    j = radial_segments
    for k in range(azimuth_segments):
        k2 = (k + 1) % azimuth_segments
        v_plane_k = plane_pts[j, k]
        v_plane_k2 = plane_pts[j, k2]
        v_asph_k = asph_pts[j, k]
        v_asph_k2 = asph_pts[j, k2]
        add_quad(v_plane_k, v_plane_k2, v_asph_k2, v_asph_k, outward_ccw=True)

    return np.asarray(tris, dtype=float)
