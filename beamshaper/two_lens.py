"""
Two-lens (Keplerian) refractive beam shaper design.

A two-element beam shaper consists of two plano-aspheric lenses:
  - Lens 1: Flat entry face, aspheric exit face → redistributes energy
  - Lens 2: Aspheric entry face, flat exit face → re-collimates

The output beam is both flat-top AND collimated, giving a depth of field
limited only by diffraction (z_R ≈ pi*R^2/lambda, typically hundreds of meters).

Design approach (Rhodes-Shealy / Hoffnagle-Jefferson):
  1. Energy conservation gives the ray mapping R(r)
  2. Snell's law at each aspheric surface gives the surface slopes
  3. The equal optical path length (OPL) condition constrains the geometry
  4. Coupled ODEs for z1(r) and z2(R) are integrated numerically

Geometry (all z measured from lens 1 flat face at z=0):
  - Lens 1: flat face at z=0, aspheric face at z = t1 - sag1(r)
  - Gap: from lens 1 exit to lens 2 entry
  - Lens 2: aspheric face, flat exit face

For a Galilean (compact) design, the gap is short and both lenses are convex.
For a Keplerian design, the beam crosses over between the lenses.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from beamshaper.aspheric import AsphericProfile


def energy_mapping(r, w0, target_radius, aperture_radius, profile_type='uniform',
                   sg_order=10, edge_width=None):
    """
    Energy conservation mapping for truncated Gaussian -> target profile.

    Supports:
    - 'uniform': hard flat-top (standard)
    - 'fermi_dirac': I(R) = 1/(1+exp((R-R0)/Delta)), Hoffnagle-Jefferson
      Smooth rolloff controlled by edge_width (default R0/10).
      Suppresses edge diffraction ripple during propagation.

    Args:
        edge_width: Fermi-Dirac rolloff width Delta (meters). Larger = softer edge.
    """
    E_r = 1.0 - np.exp(-2.0 * r ** 2 / w0 ** 2)
    E_a = 1.0 - np.exp(-2.0 * aperture_radius ** 2 / w0 ** 2)
    F_in = E_r / E_a  # normalized input CDF, 0 to 1

    if profile_type == 'uniform':
        return target_radius * np.sqrt(np.clip(F_in, 0, 1))

    elif profile_type == 'fermi_dirac':
        if edge_width is None:
            edge_width = target_radius * 0.10

        # Hoffnagle-Jefferson Fermi-Dirac profile:
        # I_out(R) = 1 / (1 + exp((R - R0) / Delta))
        #
        # Truncate at R_max = R0 + 4*Delta (where I < 2% of peak)
        # so edge rays don't map to absurd radii.
        from scipy.interpolate import interp1d
        R_max = target_radius + 4 * edge_width
        n_grid = 10000
        R_grid = np.linspace(0, R_max, n_grid)
        dR = R_grid[1] - R_grid[0]
        I_fd = 1.0 / (1.0 + np.exp((R_grid - target_radius) / edge_width))

        # CDF: integral of I(R) * R dR (power within radius R)
        integrand = I_fd * R_grid
        cdf = np.cumsum(integrand) * dR
        cdf = cdf / cdf[-1]

        cdf = np.maximum.accumulate(cdf)
        unique_mask = np.diff(cdf, prepend=-1) > 0
        inv_cdf = interp1d(cdf[unique_mask], R_grid[unique_mask],
                           bounds_error=False, fill_value=(0, R_max))
        return inv_cdf(np.clip(F_in, 0, 1))

    return target_radius * np.sqrt(np.clip(F_in, 0, 1))


def design_two_lens_shaper(
    w0: float = 40e-3,
    target_radius: float = 15e-3,
    separation: float = 100e-3,
    n_glass: float = 1.52,
    t1: float = 10e-3,
    t2: float = 10e-3,
    aperture_radius: float = 50e-3,
    n_points: int = 1000,
    profile_type: str = 'uniform',
    sg_order: int = 10,
    edge_width: float = None,
    source_distance: float = None,
) -> dict:
    """
    Design a two-lens beam shaper using the ODE approach.

    The system consists of two plano-aspheric lenses separated by a gap.
    The beam enters lens 1 through the flat face, exits through the aspheric
    face, crosses the gap, enters lens 2 through the aspheric face, and
    exits through the flat face.

    The key constraints:
    1. Energy conservation: R(r) mapping (Gaussian → uniform)
    2. Snell's law at both aspheric surfaces
    3. Collimation: output rays must be parallel to the axis
    4. Equal OPL: all rays must have the same total optical path

    For a Galilean (non-inverting) design where R(r) is monotonically
    increasing, both surfaces deflect rays outward.

    Args:
        w0: Gaussian beam waist radius
        target_radius: flat-top output beam radius
        separation: center-to-center distance between lenses
        n_glass: refractive index
        t1, t2: center thicknesses of lens 1 and lens 2
        aperture_radius: clear aperture
        n_points: number of radial sample points

    Returns:
        dict with surface profiles and fit coefficients
    """
    n_air = 1.0
    r_max = min(3.0 * w0, aperture_radius * 0.95)
    r = np.linspace(0, r_max, n_points)

    # Ray mapping (supports soft-edge profiles)
    R = energy_mapping(r, w0, target_radius, aperture_radius,
                       profile_type=profile_type, sg_order=sg_order,
                       edge_width=edge_width)

    # Geometry reference points (all z measured from lens 1 flat face):
    # Lens 1: flat face at z=0, aspheric vertex at z=t1
    # Lens 2: aspheric vertex at z=separation-t2, flat face at z=separation
    z1_vertex = t1
    z2_vertex = separation - t2

    # For each radius r, compute the required surface slopes using
    # Snell's law and the geometric constraints.
    #
    # A ray at height r enters lens 1 flat face at (r, 0), propagates
    # through glass to the aspheric surface at (r, z1_vertex - sag1(r)).
    # It refracts and travels through air to lens 2 aspheric surface
    # at (R(r), z2_vertex - sag2(R(r))), refracts again, and exits
    # lens 2 flat face at (R(r), separation) traveling parallel to z-axis.
    #
    # The slope of each surface is determined by Snell's law:
    # At surface 1: refract from glass (going +Z) to air (toward lens 2)
    # At surface 2: refract from air (coming from lens 1) to glass (going +Z)
    #
    # For collimation: the ray must exit lens 2 parallel to z,
    # so at lens 2's aspheric surface, the refracted direction must be +Z.
    # This constrains the surface 2 slope given the incoming ray angle.

    sag1 = np.zeros(n_points)
    sag2 = np.zeros(n_points)
    dsag1 = np.zeros(n_points)  # derivative tables for tabulated tracing
    dsag2 = np.zeros(n_points)

    def compute_slopes(ri, Ri, sag1_val, sag2_val):
        """
        Compute dsag1/dr and dsag2/dR at given r, R, sag values.

        For diverging input, accounts for the lateral ray shift through
        lens 1 glass: a ray entering at flat-face radius ri hits the
        aspheric face at a different radius r_asph.
        """
        z1s = z1_vertex - sag1_val
        z2s = z2_vertex - sag2_val

        if source_distance is not None and source_distance > 0:
            # Diverging source at distance d.
            # Ray from fiber hits flat face at radius ri.
            sin_air_flat = ri / np.sqrt(ri ** 2 + source_distance ** 2)
            sin_glass = sin_air_flat / n_glass
            cos_glass = np.sqrt(max(1.0 - sin_glass ** 2, 0.0))

            # Lateral shift through lens 1 glass
            glass_thickness = z1s  # from flat face (z=0) to aspheric face (z=z1s)
            r_asph = ri + glass_thickness * sin_glass / cos_glass
        else:
            sin_glass = 0.0
            cos_glass = 1.0
            r_asph = ri

        # Air ray from aspheric face of lens 1 to aspheric face of lens 2
        dx = Ri - r_asph
        dz = z2s - z1s
        norm_air = np.sqrt(dx ** 2 + dz ** 2)
        sin_air = dx / norm_air
        cos_air = dz / norm_air

        # Surface 1 slope (generalized Snell for non-zero glass angle)
        denom1 = n_air * cos_air - n_glass * cos_glass
        if abs(denom1) < 1e-15:
            s1 = 0.0
        else:
            s1 = (n_air * sin_air - n_glass * sin_glass) / denom1

        # Surface 2 slope (air → glass, output must be +Z)
        denom2 = n_glass - n_air * cos_air
        if abs(denom2) < 1e-15:
            s2 = 0.0
        else:
            s2 = -n_air * sin_air / denom2

        return s1, s2

    for i in range(1, n_points):
        ri, Ri = r[i], R[i]
        dr_val = r[i] - r[i - 1]
        dR = R[i] - R[i - 1]

        # Predictor step (Euler with previous values)
        s1_pred, s2_pred = compute_slopes(ri, Ri, sag1[i - 1], sag2[i - 1])
        sag1_pred = sag1[i - 1] + s1_pred * dr_val
        sag2_pred = sag2[i - 1] + s2_pred * dR if abs(dR) > 1e-15 else sag2[i - 1]

        # Corrector step (trapezoidal: average of old and new slopes)
        s1_corr, s2_corr = compute_slopes(ri, Ri, sag1_pred, sag2_pred)
        sag1[i] = sag1[i - 1] + 0.5 * (s1_pred + s1_corr) * dr_val
        dsag1[i] = 0.5 * (s1_pred + s1_corr)
        if abs(dR) > 1e-15:
            sag2[i] = sag2[i - 1] + 0.5 * (s2_pred + s2_corr) * dR
            dsag2[i] = 0.5 * (s2_pred + s2_corr)
        else:
            sag2[i] = sag2[i - 1]
            dsag2[i] = dsag2[i - 1] if i > 0 else 0.0

    # --- Enforce equal OPL by solving algebraically for z2 ---
    # For collimated input: OPL = n*z1 + air_gap + n*(sep-Z2)
    # For diverging input:  OPL = fiber_path + n*z1 + air_gap + n*(sep-Z2)
    # where fiber_path = sqrt(r^2 + d^2) for a ray from fiber to flat face at radius r.

    if source_distance is not None and source_distance > 0:
        # Diverging: include fiber-to-lens path
        # On-axis OPL: fiber_d + n*t1 + (sep-t1-t2) + n*t2
        C_target = source_distance + n_glass * (t1 + t2) + (separation - t1 - t2)
    else:
        # Collimated: no fiber path
        C_target = n_glass * (t1 + t2) + (separation - t1 - t2)

    sag2_opl = np.zeros(n_points)
    for i in range(n_points):
        z = z1_vertex - sag1[i]  # z-position of lens 1 aspheric surface
        ri, Ri = r[i], R[i]

        # Fiber path (air, from fiber to lens 1 flat face)
        if source_distance is not None and source_distance > 0:
            fiber_opl = np.sqrt(ri ** 2 + source_distance ** 2)
        else:
            fiber_opl = 0.0

        # Remaining OPL budget after fiber path and lens 1 glass:
        # C = fiber_opl + n*z + air_gap_length + n*(sep - Z2)
        # air_gap_length = sqrt((Ri - ri_asph)^2 + (Z2 - z)^2)
        # Solve for Z2:
        # C - fiber_opl - n*z - n*sep = air_gap - n*Z2
        # Let Rem = C - fiber_opl - n*z - n*sep  (note: Rem < 0 typically)
        # sqrt((Ri-ri_asph)^2 + (Z2-z)^2) = Rem + n*Z2

        # Account for lateral shift through lens 1 glass
        if source_distance is not None and source_distance > 0:
            sin_air_flat = ri / np.sqrt(ri ** 2 + source_distance ** 2)
            sin_g = sin_air_flat / n_glass
            cos_g = np.sqrt(max(1.0 - sin_g ** 2, 0))
            ri_asph = ri + z * sin_g / cos_g if cos_g > 1e-10 else ri
        else:
            ri_asph = ri

        Rem = C_target - fiber_opl - n_glass * z - n_glass * separation
        # (Ri-ri_asph)^2 + (Z2-z)^2 = (Rem + n*Z2)^2
        # Expand: (Ri-ri_asph)^2 + Z2^2 - 2*Z2*z + z^2 = Rem^2 + 2*Rem*n*Z2 + n^2*Z2^2
        a_coeff = 1.0 - n_glass ** 2
        b_coeff = -2.0 * (z + Rem * n_glass)
        c_coeff = (Ri - ri_asph) ** 2 + z ** 2 - Rem ** 2

        disc = b_coeff ** 2 - 4 * a_coeff * c_coeff
        if disc >= 0:
            Z1 = (-b_coeff + np.sqrt(disc)) / (2 * a_coeff)
            Z2 = (-b_coeff - np.sqrt(disc)) / (2 * a_coeff)
            s1_cand = z2_vertex - Z1
            s2_cand = z2_vertex - Z2
            sag2_opl[i] = s1_cand if abs(s1_cand - sag2[i]) < abs(s2_cand - sag2[i]) else s2_cand
        else:
            sag2_opl[i] = sag2[i]
    sag2 = sag2_opl

    # --- Reparameterize sag1 from flat-face radius to aspheric-face radius ---
    # For diverging input, the ray at flat-face radius r hits the aspheric face
    # at r_asph = r + glass_thickness * tan(theta_glass). The tracer intersects
    # the surface at r_asph, so the sag table must be indexed by r_asph.
    if source_distance is not None and source_distance > 0:
        r_asph = np.zeros(n_points)
        for i in range(n_points):
            sin_air_flat = r[i] / np.sqrt(r[i] ** 2 + source_distance ** 2)
            sin_g = sin_air_flat / n_glass
            cos_g = np.sqrt(max(1.0 - sin_g ** 2, 0.0))
            z1s = z1_vertex - sag1[i]
            r_asph[i] = r[i] + z1s * sin_g / cos_g if cos_g > 1e-10 else r[i]
        # Resample sag1 onto a regular grid in r_asph space
        r_asph_regular = np.linspace(0, r_asph[-1], n_points)
        sag1 = np.interp(r_asph_regular, r_asph, sag1)
        r = r_asph_regular  # r now represents aspheric-face radius

    # Recompute derivatives after reparameterization
    dsag1 = np.gradient(sag1, r)
    dsag2 = np.gradient(sag2, R)

    # OPL verification
    opl = np.zeros(n_points)
    for i in range(n_points):
        z1s = z1_vertex - sag1[i]
        z2s = z2_vertex - sag2[i]
        dx = R[i] - r[i]
        dz = z2s - z1s
        opl[i] = n_glass * z1s + np.sqrt(dx**2 + dz**2) + n_glass * (separation - z2s)

    opl_var = (np.max(opl) - np.min(opl)) / np.mean(opl) * 100
    print(f"  OPL variation: {opl_var:.6f}%")

    # Fit both surfaces to even asphere polynomials (kept for backward compat)
    coeffs1 = _fit_sag(r, sag1, n_coeffs=8)
    coeffs2 = _fit_sag(R, sag2, n_coeffs=8)

    return {
        'r': r,
        'R': R,
        'sag1': sag1,
        'sag2': sag2,
        'dsag1': dsag1,
        'dsag2': dsag2,
        'opl': opl,
        'separation': separation,
        't1': t1,
        't2': t2,
        'coeffs1': coeffs1,
        'coeffs2': coeffs2,
        'source_distance': source_distance,
    }


def _fit_sag(r, sag, n_coeffs=8):
    """Fit sag to even asphere: c*r²/(1+sqrt(1-(1+k)*c²r²)) + poly."""
    r_nz = r[r > 0]
    sag_nz = sag[r > 0]
    if len(r_nz) < 3:
        return {'curvature': 0.0, 'conic': 0.0, 'alphas': np.zeros(n_coeffs)}

    # Initial curvature from parabolic fit
    idx = max(2, len(r_nz) // 10)
    c_init = 2.0 * np.mean(sag_nz[:idx] / (r_nz[:idx] ** 2 + 1e-30))

    def sag_model(params, r):
        c, k = params[0], params[1]
        alphas = params[2:]
        r2 = r ** 2
        denom_arg = np.maximum(1.0 - (1.0 + k) * c * c * r2, 1e-30)
        z = c * r2 / (1.0 + np.sqrt(denom_arg))
        rp = r2 * r2
        for a in alphas:
            z = z + a * rp
            rp = rp * r2
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
