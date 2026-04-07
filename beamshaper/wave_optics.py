"""
Scalar wave optics beam propagation for beam shaping verification.

Uses a 1D radial simulation (cylindrical symmetry) with the Collins integral
(ABCD matrix formalism). The lens is decomposed into thin-lens + residual to
reduce the oscillation frequency in the integrand, and the Collins integral
is computed as a vectorized matrix operation for speed.

The input grid must be fine enough to resolve the lens phase gradient:
    dx < lambda / (2 * (n-1) * max(dsag/dr))

For typical parameters (lambda=780nm, n=1.52, dsag/dr~0.25), dx < 3um,
requiring ~20,000 radial points over a 60mm aperture.
"""

import numpy as np
from scipy.special import j0
from beamshaper.aspheric import AsphericProfile, aspheric_sag, aspheric_deriv


def simulate_beam_shaping(
    profile: AsphericProfile,
    center_thickness: float = 50e-3,
    n_glass: float = 1.52,
    wavelength: float = 780e-9,
    w0: float = 40e-3,
    waist_z: float = -0.10,
    launch_z: float = -0.06,
    target_z: float = 0.25,
    n_radial_in: int = 20000,
    n_radial_out: int = 500,
    r_max_out: float = None,
) -> dict:
    """
    1D radial wave optics simulation via Collins integral.

    Decomposes the lens into thin-lens (f_eff chosen so A=0 in ABCD matrix)
    plus residual, then computes the Collins integral as a vectorized
    matrix-vector product.
    """
    if r_max_out is None:
        r_max_out = profile.aperture_radius * 0.6

    r_max_in = profile.aperture_radius * 1.1
    r_in = np.linspace(0, r_max_in, n_radial_in)
    r_out = np.linspace(0, r_max_out, n_radial_out)
    dr = np.zeros(n_radial_in)
    dr[1:] = np.diff(r_in)
    dr[0] = r_in[1] if n_radial_in > 1 else 1.0

    k = 2.0 * np.pi / wavelength

    # --- Check grid resolution ---
    max_slope = float(np.max(np.abs(aspheric_deriv(
        r_in[r_in <= profile.aperture_radius], profile))))
    dx = r_in[1] - r_in[0]
    dx_required = wavelength / (2.0 * (n_glass - 1.0) * max_slope) if max_slope > 0 else 1.0
    print(f"  Grid: dx={dx*1e6:.1f}um, required<{dx_required*1e6:.1f}um", end="")
    if dx > dx_required:
        print(f" WARNING: undersampled! Increase n_radial_in to >{int(r_max_in/dx_required)}")
    else:
        print(f" OK ({dx/dx_required:.2f}x Nyquist)")

    # --- Gaussian beam at lens plane ---
    # Fresnel number for launch->lens is huge (~34,000), so diffraction
    # over 60mm is negligible. Use the geometric Gaussian directly at the lens.
    zR = np.pi * w0 ** 2 / wavelength
    z_from_waist = -waist_z  # distance from waist to lens (at z=0)
    w_at_lens = w0 * np.sqrt(1.0 + (z_from_waist / zR) ** 2)

    E_at_lens = np.exp(-r_in ** 2 / w_at_lens ** 2).astype(complex)
    if abs(z_from_waist) > 1e-15:
        R_wf = z_from_waist * (1.0 + (zR / z_from_waist) ** 2)
        E_at_lens *= np.exp(-1j * k * r_in ** 2 / (2.0 * R_wf))

    # --- Lens phase ---
    sag_vals = aspheric_sag(r_in, profile)
    thickness = center_thickness - sag_vals
    phi_total = k * (n_glass - 1.0) * thickness
    phi_total -= phi_total[0]  # remove piston

    aperture = (r_in <= profile.aperture_radius).astype(float)

    # --- Decompose: choose f_eff = d_prop so A=0 in Collins matrix ---
    d_prop = target_z  # lens at z=0, target at z=target_z
    f_eff = d_prop  # makes A = 1 - d/f = 0

    phi_thin = -k * r_in ** 2 / (2.0 * f_eff)
    phi_residual = phi_total - phi_thin
    phi_residual -= phi_residual[0]  # remove piston

    res_waves = np.ptp(phi_residual[r_in <= profile.aperture_radius]) / (2 * np.pi)
    print(f"  f_eff={f_eff*1e3:.1f}mm (= target_z), residual: {res_waves:.1f} waves")

    # ABCD matrix: thin lens (f_eff) + free space (d_prop)
    A = 0.0  # by construction
    B = d_prop
    D = 1.0

    # --- Apply residual phase and aperture ---
    E_with_phase = E_at_lens * aperture * np.exp(1j * phi_residual)

    # --- Vectorized Collins integral ---
    # E_out(r2) = (-ik/B) * sum_i [ E'(r1_i) * exp(ikA*r1_i^2/(2B)) * J0(k*r1_i*r2/B) * r1_i * dr_i ]
    # With A=0, the quadratic phase term vanishes!

    prefactor = -1j * k / B

    integrand_base = E_with_phase * r_in * dr  # (N_in,)

    # Compute J0 matrix in chunks to manage memory
    print(f"  Computing Collins integral ({n_radial_in}x{n_radial_out})...")
    chunk_size = 50  # output points per chunk
    E_target = np.zeros(n_radial_out, dtype=complex)

    for start in range(0, n_radial_out, chunk_size):
        end = min(start + chunk_size, n_radial_out)
        r_chunk = r_out[start:end]

        # J0 matrix: (N_in, chunk_size)
        bessel_arg = (k / B) * r_in[:, None] * r_chunk[None, :]
        J0_matrix = j0(bessel_arg)

        # Integrate
        E_chunk = integrand_base[:, None] * J0_matrix
        E_target[start:end] = prefactor * np.sum(E_chunk, axis=0)

    # Output quadratic phase (D*r2^2 term)
    E_target *= np.exp(1j * k * D * r_out ** 2 / (2.0 * B))

    I_target = np.abs(E_target) ** 2
    I_target_norm = I_target / np.max(I_target) if np.max(I_target) > 0 else I_target

    I_input = np.abs(E_at_lens) ** 2
    I_input /= np.max(I_input)

    return {
        'r_in': r_in,
        'r_out': r_out,
        'I_input': I_input,
        'I_target': I_target_norm,
        'I_target_raw': I_target,
        'E_target': E_target,
        'f_eff': f_eff,
        'phi_residual': phi_residual,
    }
