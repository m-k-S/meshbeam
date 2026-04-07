"""
Analytical aspheric beam shaper design via ray mapping + Snell's law.

Instead of optimizing polynomial coefficients with gradient descent,
this module computes the aspheric surface profile directly from:
1. Energy conservation: R(r) = R_0 * sqrt(1 - exp(-2r^2/w0^2))
2. Snell's law at the aspheric surface
3. Geometric ray targeting to the output plane

The surface sag is obtained by integrating dz/dr from the required
refraction angles, then fit to an even asphere polynomial for use
in the optimizer and mesh generator.

References:
    - Shealy & Hoffnagle, SPIE 5876 (2005)
    - Kreuzer, "Coherent light optical system yielding an output beam
      of desired intensity distribution at a desired equiphase surface"
    - Dickey & Holswade, "Laser Beam Shaping" (2000)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares


def gaussian_to_uniform_mapping(r, w0, target_radius, aperture_radius=None):
    """Energy conservation mapping: truncated Gaussian intensity -> uniform."""
    if aperture_radius is None:
        aperture_radius = 3.0 * w0
    E_r = 1.0 - np.exp(-2.0 * r ** 2 / w0 ** 2)
    E_a = 1.0 - np.exp(-2.0 * aperture_radius ** 2 / w0 ** 2)
    return target_radius * np.sqrt(E_r / E_a)


def compute_surface_profile(
    w0: float,
    target_radius: float,
    target_z: float,
    n_glass: float,
    center_thickness: float,
    aperture_radius: float,
    n_points: int = 500,
) -> tuple:
    """
    Compute the aspheric surface sag profile from ray-mapping equations.

    The beam enters the flat face at z = -ct/2, traverses the glass,
    and exits through the aspheric face at z = ct/2 - sag(r).
    After exiting, each ray propagates to the target plane at z = target_z,
    arriving at radius R(r) determined by energy conservation.

    The required surface slope is derived from Snell's law: given the
    glass-side ray direction and the desired air-side direction, the
    surface normal (and hence slope) is determined.

    Args:
        w0: Gaussian beam waist radius at the lens
        target_radius: desired flat-top output radius
        target_z: z-position of the target plane
        n_glass: refractive index of glass
        center_thickness: lens center thickness
        aperture_radius: lens clear aperture radius
        n_points: number of radial sample points

    Returns:
        r: (N,) radial positions
        sag: (N,) surface sag values (z_surface = z_vertex - sag)
        dz_dr: (N,) surface slope
        R_target: (N,) target output radii
    """
    ct = center_thickness
    z_flat = -ct / 2.0
    z_vertex = ct / 2.0

    # Use 3*w0 as the effective beam radius (99.7% of energy)
    r_max = min(3.0 * w0, aperture_radius * 0.95)
    r = np.linspace(0, r_max, n_points)

    # Target output radii from energy conservation (truncated Gaussian)
    R_target = gaussian_to_uniform_mapping(r, w0, target_radius, aperture_radius)

    # For each radial position, compute the required surface slope.
    #
    # The ray enters the flat face at height r (approximately, for
    # a collimated beam hitting a flat surface, no deflection).
    # Inside glass, the ray travels in +Z at height r.
    #
    # At the aspheric surface z(r) = z_vertex - sag(r), the ray
    # must be refracted so that it arrives at height R(r) at z = target_z.
    #
    # After refraction, the ray travels from (r, z_surface) to (R, target_z)
    # in a straight line:
    #   direction_air = (R(r) - r, 0, target_z - z_surface) normalized
    #
    # But z_surface depends on sag, which we're computing. We integrate
    # dz/dr forward using the refraction condition.

    sag = np.zeros(n_points)
    dz_dr = np.zeros(n_points)

    # At r=0: by symmetry, the surface is flat (dz/dr = 0), and the
    # ray goes straight through (R(0) = 0).
    for i in range(1, n_points):
        ri = r[i]
        Ri = R_target[i]

        # Current z of aspheric surface
        z_surf = z_vertex - sag[i - 1]  # approximate with previous sag

        # Required output direction in air (from surface to target)
        dx_air = Ri - ri  # transverse displacement
        dz_air = target_z - z_surf  # axial distance

        # Normalize to get air-side direction
        norm_air = np.sqrt(dx_air ** 2 + dz_air ** 2)
        sin_air = dx_air / norm_air  # sin of angle with z-axis
        cos_air = dz_air / norm_air

        # Glass-side direction: straight +Z (collimated in glass after flat face)
        sin_glass = 0.0
        cos_glass = 1.0

        # From Snell's law in the tangent plane:
        # n_glass * sin(theta_glass_from_normal) = n_air * sin(theta_air_from_normal)
        #
        # For a surface with slope angle alpha (dz/dr = tan(alpha)):
        #   theta_glass_from_normal = alpha (since glass ray is along Z)
        #   theta_air_from_normal can be derived from the air direction and normal
        #
        # More directly: the surface normal direction determines the refraction.
        # We can find the required surface normal from Snell's law in vector form.
        #
        # Glass direction: d_g = (0, 0, 1)  [propagating in +Z through glass]
        # Air direction: d_a = (sin_air, 0, cos_air)
        #
        # Snell's law (vector form):
        #   n_glass * (d_g x n_hat) = n_air * (d_a x n_hat)
        #
        # For 2D (r-z plane), the surface normal is n_hat = (-sin_alpha, 0, cos_alpha)
        # where alpha is the surface tilt angle (dz/dr = -tan(alpha) for sag convention).
        #
        # Using the scalar Snell's law with proper angles:
        # The angle of incidence (glass side, from surface normal):
        #   theta_i = alpha (surface tilted by alpha from Z-axis normal)
        # The angle of refraction (air side, from surface normal):
        #   Using Snell: n_glass * sin(theta_i) = sin(theta_r)
        #
        # But we need to solve for alpha given the required deflection.
        # Use the refraction geometry directly:
        #
        # The transverse momentum change at the surface:
        #   n_glass * sin(angle_glass_with_z) - n_air * sin(angle_air_with_z)
        #     = (n_glass - n_air) * sin(surface_tilt)  ... not exactly right
        #
        # Let me use the exact vector approach.

        # We need to find surface slope dz/dr such that:
        # A ray going (0, 1) in glass [r-direction is 0, z-direction is 1]
        # refracts to (sin_air, cos_air) in air.
        #
        # Surface normal (outward, pointing +Z at vertex):
        #   n_hat = (dz_dr_val, 1) / norm  [in the r-z plane, outward]
        #   Here we use the convention: surface is z = z_vertex - sag(r),
        #   so F = z - z_vertex + sag(r) = 0, gradient = (dsag/dr, 1).
        #
        # Apply Snell's law at this surface.
        # The key relation: for refraction with surface normal n_hat,
        # the tangential component of (n*direction) is preserved:
        #
        #   n_glass * (d_glass - (d_glass . n_hat) * n_hat)
        #     = n_air * (d_air - (d_air . n_hat) * n_hat)
        #
        # In 2D (r, z) with d_glass = (0, 1), d_air = (sin_air, cos_air):
        # n_hat = (s, 1)/sqrt(s^2+1) where s = dsag/dr
        #
        # d_glass . n_hat = 1/sqrt(s^2+1)
        # d_air . n_hat = (sin_air * s + cos_air)/sqrt(s^2+1)
        #
        # The tangential component preservation gives us an equation in s.
        # But it's cleaner to use the cross-product form of Snell's law:
        #
        # n_glass * |d_glass x n_hat| = n_air * |d_air x n_hat|
        #   ... no, this gives magnitudes. Use signed version:
        #
        # n_glass * (d_glass x n_hat) = n_air * (d_air x n_hat)
        #
        # In 2D, cross product gives a scalar:
        # d_glass x n_hat = (0)(1/N) - (1)(s/N) = -s/N
        # d_air x n_hat = (sin_air)(1/N) - (cos_air)(s/N)
        #               = (sin_air - cos_air * s) / N
        #
        # where N = sqrt(s^2 + 1)
        #
        # Snell: n_glass * (-s/N) = n_air * (sin_air - cos_air*s) / N
        # The N cancels:
        # -n_glass * s = n_air * sin_air - n_air * cos_air * s
        # -n_glass * s + n_air * cos_air * s = n_air * sin_air
        # s * (n_air * cos_air - n_glass) = n_air * sin_air
        # s = n_air * sin_air / (n_air * cos_air - n_glass)

        n_air = 1.0
        slope = n_air * sin_air / (n_air * cos_air - n_glass)

        dz_dr[i] = slope

        # Integrate sag: sag(r) = integral of dz_dr from 0 to r
        dr = r[i] - r[i - 1]
        sag[i] = sag[i - 1] + slope * dr

    return r, sag, dz_dr, R_target


def fit_aspheric_coefficients(
    r: np.ndarray,
    sag: np.ndarray,
    n_coeffs: int = 6,
) -> dict:
    """
    Fit the computed sag profile to an even asphere polynomial.

    z(r) = c*r^2/(1+sqrt(1-(1+k)*c^2*r^2)) + sum(alpha_i * r^(2i+4))

    Returns dict with 'curvature', 'conic', 'alphas'.
    """
    # Initial guess: fit a parabola for the curvature
    # sag ≈ c/2 * r^2 for small r → c ≈ 2*sag/r^2
    r_nz = r[r > 0]
    sag_nz = sag[r > 0]

    if len(r_nz) < 3:
        return {'curvature': 0.0, 'conic': 0.0, 'alphas': np.zeros(n_coeffs)}

    # Fit: curvature from parabolic term
    # For small r: sag ≈ c*r^2/2
    idx_small = len(r_nz) // 10  # use first 10% of data
    if idx_small < 2:
        idx_small = 2
    c_init = 2.0 * np.mean(sag_nz[:idx_small] / (r_nz[:idx_small] ** 2 + 1e-30))

    def sag_model(params, r):
        c = params[0]
        k = params[1]
        alphas = params[2:]

        r2 = r ** 2
        denom_arg = 1.0 - (1.0 + k) * c * c * r2
        denom_arg = np.maximum(denom_arg, 1e-30)
        z = c * r2 / (1.0 + np.sqrt(denom_arg))

        r_power = r2 * r2
        for a in alphas:
            z = z + a * r_power
            r_power = r_power * r2

        return z

    def residuals(params):
        return sag_model(params, r_nz) - sag_nz

    x0 = np.zeros(2 + n_coeffs)
    x0[0] = c_init

    result = least_squares(residuals, x0, method='lm')

    return {
        'curvature': result.x[0],
        'conic': result.x[1],
        'alphas': result.x[2:],
    }


def design_beam_shaper(
    w0: float = 40e-3,
    target_radius: float = 15e-3,
    target_z: float = 0.25,
    n_glass: float = 1.52,
    center_thickness: float = 50e-3,
    aperture_radius: float = 50e-3,
    n_alpha_coeffs: int = 6,
) -> dict:
    """
    Full analytical beam shaper design.

    Computes the aspheric surface profile and fits it to an even asphere.

    Returns:
        dict with 'curvature', 'conic', 'alphas', 'r', 'sag', 'R_target'
    """
    r, sag, dz_dr, R_target = compute_surface_profile(
        w0=w0,
        target_radius=target_radius,
        target_z=target_z,
        n_glass=n_glass,
        center_thickness=center_thickness,
        aperture_radius=aperture_radius,
    )

    coeffs = fit_aspheric_coefficients(r, sag, n_coeffs=n_alpha_coeffs)

    return {
        **coeffs,
        'r': r,
        'sag': sag,
        'dz_dr': dz_dr,
        'R_target': R_target,
    }
