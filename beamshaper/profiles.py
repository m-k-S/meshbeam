"""
Beam shaping target profiles and loss functions.

Implements the Zemax-style ray-mapping approach: each input ray at radius r
gets a target output radius R(r) derived from energy conservation.

For Gaussian input -> uniform output, the mapping is:
    R(r) = R_0 * sqrt(1 - exp(-2r^2/w_0^2))

The loss function is simply sum((actual_R_i - target_R_i)^2), which gives
strong, direct gradients to the optimizer (equivalent to Zemax REAY operands).

References:
    - Shealy & Hoffnagle, "Laser beam shaping profiles and propagation"
    - Dickey, "Laser Beam Shaping: Theory and Techniques"
    - Zemax Knowledge Base: "How to design a Gaussian to Top Hat beam shaper"
"""

import jax
import jax.numpy as jnp


def gaussian_to_uniform_mapping(r_input, w0, target_radius, aperture_radius=None):
    """
    Ray-mapping function from energy conservation for a truncated Gaussian.

    With intensity-proportional sampling (sigma = w0/2), each ray carries
    equal power. The energy CDF is E(r) = 1 - exp(-2r^2/w0^2). For uniform
    irradiance at the output:

        R(r) = R_0 * sqrt(E(r) / E(a))

    where E(a) accounts for truncation at the aperture radius a.

    Args:
        r_input: (N,) input radial positions
        w0: Gaussian beam waist radius (1/e^2 intensity)
        target_radius: desired flat-top output radius R_0
        aperture_radius: beam truncation radius (if None, uses 3*w0)

    Returns:
        (N,) target output radial positions
    """
    if aperture_radius is None:
        aperture_radius = 3.0 * w0

    E_r = 1.0 - jnp.exp(-2.0 * r_input ** 2 / w0 ** 2)
    E_a = 1.0 - jnp.exp(-2.0 * aperture_radius ** 2 / w0 ** 2)

    return target_radius * jnp.sqrt(E_r / E_a)


def ray_mapping_loss(positions, weights, input_radii, w0, target_radius,
                     aperture_radius=None):
    """
    Zemax-style ray-targeting loss function.

    For each ray, compares its actual output radius to the energy-conservation
    target radius. This is equivalent to Zemax's REAY merit function operands.

    L = (1/N) * sum_i w_i * (|pos_i| - R(r_i))^2

    Args:
        positions: (N, 2) ray positions at target plane
        weights: (N,) intensity weights (aperture/TIR)
        input_radii: (N,) input radial positions of rays at the lens
        w0: Gaussian beam waist at the lens
        target_radius: desired flat-top output radius
        aperture_radius: beam truncation radius

    Returns:
        scalar loss value
    """
    # Target output radii from energy conservation (truncated Gaussian)
    target_r = gaussian_to_uniform_mapping(input_radii, w0, target_radius, aperture_radius)

    # Actual output radii
    actual_r = jnp.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)

    # Weighted MSE between actual and target radii
    residuals = (actual_r - target_r) ** 2
    loss = jnp.sum(weights * residuals) / (jnp.sum(weights) + 1e-30)

    return loss


# Keep the old functions around for the visualization (radial profile plot)

def flat_top_target(r, target_radius, edge_steepness=50.0):
    """Smooth flat-top target intensity profile for visualization."""
    return jax.nn.sigmoid((target_radius - r) * edge_steepness)


def radial_kde(r_samples, weights, r_eval, bandwidth):
    """1D KDE for visualization of the radial intensity profile."""
    diff = r_eval[:, None] - r_samples[None, :]
    kernels = jnp.exp(-0.5 * (diff / bandwidth) ** 2)
    density = jnp.sum(kernels * weights[None, :], axis=1)
    density = density / (jnp.max(density) + 1e-30)
    return density
