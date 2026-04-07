"""
Gradient-descent optimization of aspheric lens parameters for beam shaping.

Uses the Zemax-style ray-mapping approach: each input ray gets a target
output radius derived from energy conservation (Gaussian -> uniform mapping).
The loss is simply sum((actual_R - target_R)^2), giving strong direct gradients.

References:
    - Zemax KB: "How to design a Gaussian to Top Hat beam shaper"
    - Shealy & Hoffnagle, SPIE 5876 (2005)
"""

import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List
from beamshaper.jax_tracer import trace_bundle, gaussian_beam_rays_jax
from beamshaper.profiles import ray_mapping_loss


@dataclass
class BeamShapingConfig:
    """Configuration for beam shaping optimization."""
    # Beam parameters
    n_rays: int = 500
    waist_radius: float = 40e-3     # w0 in meters
    wavelength: float = 780e-9      # meters
    waist_z: float = -0.10          # beam waist z-position
    launch_z: float = -0.06         # ray launch plane

    # Lens parameters (initial values)
    curvature: float = 6.25         # 1/R, R=160mm
    conic_constant: float = 0.0
    n_alpha_coeffs: int = 4         # number of polynomial coefficients
    center_thickness: float = 50e-3
    n_glass: float = 1.52
    aperture_radius: float = 50e-3

    # Target
    target_z: float = 0.20          # target plane z-position (beyond lens)
    target_radius: float = 20e-3    # desired flat-top radius

    # Optimization
    learning_rate: float = 1e-2
    max_iterations: int = 300
    seed: int = 42


@dataclass
class OptimizationResult:
    """Result of beam shaping optimization."""
    final_curvature: float
    final_conic: float
    final_alphas: Any
    loss_history: List[float]
    param_history: List[Dict[str, Any]]
    final_positions: Any
    final_weights: Any
    converged: bool


def make_loss_fn(origins, directions, config: BeamShapingConfig):
    """
    Create a loss function using the ray-mapping approach.

    Each ray's actual output radius is compared to its energy-conservation
    target radius R(r) = R_0 * sqrt(1 - exp(-2r^2/w0^2)).
    """
    # w0 at the lens entry face: the beam is nearly collimated (huge Rayleigh range),
    # so w at the lens is approximately w0.
    # More precisely, compute w at z=0 (lens position):
    w0 = config.waist_radius
    zR = jnp.pi * w0 ** 2 / config.wavelength
    z_lens = 0.0 - config.waist_z  # distance from waist to lens
    w_at_lens = w0 * jnp.sqrt(1.0 + (z_lens / zR) ** 2)

    def loss_fn(params):
        c = params['curvature']
        k = params['conic']
        alphas = params['alphas']

        positions, weights, input_radii = trace_bundle(
            origins, directions,
            c, k, alphas,
            config.center_thickness,
            config.n_glass,
            config.aperture_radius,
            config.target_z,
        )

        return ray_mapping_loss(
            positions, weights, input_radii,
            w0=w_at_lens,
            target_radius=config.target_radius,
            aperture_radius=config.aperture_radius,
        )

    return loss_fn


def optimize(config: BeamShapingConfig, callback: Optional[Callable] = None, stop_flag: Optional[Callable] = None) -> OptimizationResult:
    """
    Run beam shaping optimization.

    Args:
        config: optimization configuration
        callback: optional fn(iteration, loss, params, positions, weights)
            called each iteration for live UI updates
        stop_flag: optional fn() -> bool, returns True to stop early

    Returns:
        OptimizationResult
    """
    # Generate fixed ray set
    key = jax.random.PRNGKey(config.seed)
    origins, directions = gaussian_beam_rays_jax(
        config.n_rays, config.waist_radius, config.wavelength,
        config.waist_z, config.launch_z, key,
    )

    # Initial parameters
    params = {
        'curvature': jnp.array(config.curvature),
        'conic': jnp.array(config.conic_constant),
        'alphas': jnp.zeros(config.n_alpha_coeffs),
    }

    # Build loss and gradient functions
    loss_fn = make_loss_fn(origins, directions, config)
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Optimizer: Adam
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    loss_history = []
    param_history = []

    for i in range(config.max_iterations):
        if stop_flag is not None and stop_flag():
            break

        loss_val, grads = value_and_grad_fn(params)
        loss_val = float(loss_val)
        loss_history.append(loss_val)

        # Record parameters
        param_history.append({
            'curvature': float(params['curvature']),
            'conic': float(params['conic']),
            'alphas': [float(a) for a in params['alphas']],
        })

        # Update
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if callback is not None:
            positions, weights, _ = trace_bundle(
                origins, directions,
                params['curvature'], params['conic'], params['alphas'],
                config.center_thickness, config.n_glass,
                config.aperture_radius, config.target_z,
            )
            callback(i, loss_val, params, positions, weights)

    # Final trace
    final_positions, final_weights, _ = trace_bundle(
        origins, directions,
        params['curvature'], params['conic'], params['alphas'],
        config.center_thickness, config.n_glass,
        config.aperture_radius, config.target_z,
    )

    return OptimizationResult(
        final_curvature=float(params['curvature']),
        final_conic=float(params['conic']),
        final_alphas=params['alphas'],
        loss_history=loss_history,
        param_history=param_history,
        final_positions=final_positions,
        final_weights=final_weights,
        converged=len(loss_history) == config.max_iterations,
    )
