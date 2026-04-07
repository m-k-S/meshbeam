"""
Differentiable ray tracer for plano-aspheric lenses using JAX.

Traces rays through the analytic aspheric surface equation (no mesh),
enabling automatic differentiation for optimization.

Conventions (matching mesh-optics-master):
- Z is the propagation axis
- Aspheric face at +Z side (toward incoming beam), flat face at -Z side
- Beam propagates in +Z direction toward the lens
- Normals point outward from the lens
"""

import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# Aspheric sag and derivatives (JAX-compatible)
# ---------------------------------------------------------------------------

def sag(r, c, k, alphas):
    """
    Even aspheric sag: z(r) = c*r^2/(1+sqrt(1-(1+k)*c^2*r^2)) + poly(r).

    All args are scalars or JAX arrays. alphas is a 1D array of coefficients
    for r^4, r^6, r^8, ...
    """
    r2 = r * r
    denom_arg = 1.0 - (1.0 + k) * c * c * r2
    denom_arg = jnp.maximum(denom_arg, 1e-30)
    z = c * r2 / (1.0 + jnp.sqrt(denom_arg))

    # Polynomial terms
    r_power = r2 * r2  # r^4
    for i in range(alphas.shape[0]):
        z = z + alphas[i] * r_power
        r_power = r_power * r2

    return z


def sag_deriv(r, c, k, alphas):
    """Analytic dz/dr of the aspheric sag (for surface normals)."""
    r2 = r * r
    denom_arg = 1.0 - (1.0 + k) * c * c * r2
    denom_arg = jnp.maximum(denom_arg, 1e-30)
    dz = c * r / jnp.sqrt(denom_arg)

    r_power = r2 * r  # r^3
    for i in range(alphas.shape[0]):
        exponent = 2 * i + 4
        dz = dz + alphas[i] * exponent * r_power
        r_power = r_power * r2

    return dz


# ---------------------------------------------------------------------------
# Ray-surface intersection
# ---------------------------------------------------------------------------

def _ray_aspheric_intersect(origin, direction, c, k, alphas, z_vertex, n_iter=30):
    """
    Find intersection of a ray with the aspheric surface z = z_vertex - sag(r).

    For a convex surface (c > 0), the surface drops below z_vertex at larger r.
    Uses Newton's method with fixed iteration count (differentiable).
    Returns (t, hit_point) where hit_point = origin + t * direction.
    """
    # Initial guess: intersection with tangent plane at vertex (z = z_vertex)
    t = (z_vertex - origin[2]) / (direction[2] + 1e-30)

    def newton_step(t, _):
        p = origin + t * direction
        r = jnp.sqrt(p[0] ** 2 + p[1] ** 2)
        # f(t) = p_z - (z_vertex - sag(r)) = p_z - z_vertex + sag(r)
        f = p[2] - z_vertex + sag(r, c, k, alphas)
        # df/dt = dir_z + dsag/dr * dr/dt
        dr_dt = (p[0] * direction[0] + p[1] * direction[1]) / (r + 1e-30)
        df_dt = direction[2] + sag_deriv(r, c, k, alphas) * dr_dt
        t = t - f / (df_dt + 1e-30)
        return t, None

    t, _ = jax.lax.scan(newton_step, t, None, length=n_iter)

    hit = origin + t * direction
    return t, hit


def _surface_normal_aspheric(point, c, k, alphas, z_vertex):
    """
    Outward normal (pointing +Z at vertex) at a point on the aspheric surface.

    Surface: z = z_vertex - sag(r), so F = z - z_vertex + sag(r) = 0.
    Gradient: (dsag/dr * cos_t, dsag/dr * sin_t, 1) = outward normal.
    """
    x, y = point[0], point[1]
    r = jnp.sqrt(x ** 2 + y ** 2)
    dz_dr = sag_deriv(r, c, k, alphas)

    cos_t = x / (r + 1e-30)
    sin_t = y / (r + 1e-30)

    nx = dz_dr * cos_t
    ny = dz_dr * sin_t
    nz = 1.0

    norm = jnp.sqrt(nx * nx + ny * ny + nz * nz)
    return jnp.array([nx / norm, ny / norm, nz / norm])


# ---------------------------------------------------------------------------
# Snell's law (differentiable)
# ---------------------------------------------------------------------------

def _snell_refract(direction, normal, n1, n2):
    """
    Snell's law refraction. Normal must point INTO the destination medium (n2).

    Returns (refracted_direction, weight) where weight is 0 for TIR
    and 1 for normal refraction (with smooth transition).
    """
    d = direction / (jnp.linalg.norm(direction) + 1e-30)
    n = normal / (jnp.linalg.norm(normal) + 1e-30)

    cosi = -jnp.dot(d, n)
    cosi = jnp.clip(cosi, -1.0, 1.0)

    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)

    # Smooth TIR handling: sigmoid transition near k=0
    weight = jax.nn.sigmoid(k * 1e4)
    k_safe = jnp.maximum(k, 1e-8)

    t = eta * d + (eta * cosi - jnp.sqrt(k_safe)) * n
    t = t / (jnp.linalg.norm(t) + 1e-30)

    return t, weight


# ---------------------------------------------------------------------------
# Full ray trace through plano-aspheric lens
# ---------------------------------------------------------------------------

def trace_ray(origin, direction, c, k, alphas, center_thickness, n_glass, aperture_radius, target_z):
    """
    Trace a single ray through a plano-aspheric lens to a target plane.

    Lens geometry (matching mesh-optics-master convention):
    - Flat face at z = -center_thickness/2  (entry side, facing -Z / incoming beam)
    - Aspheric face at z = +center_thickness/2 + sag(r)  (exit side, convex toward +Z)

    The beam travels in +Z, enters the flat face first, refracts into glass,
    hits the aspheric exit face, refracts back to air, then propagates
    to the target plane at z = target_z (should be > center_thickness/2).

    Returns (target_xy, weight) where target_xy is the (x,y) position
    at the target plane and weight accounts for aperture clipping and TIR.
    """
    z_flat = -center_thickness * 0.5  # flat entry face
    z_asph = center_thickness * 0.5   # aspheric vertex (exit side)

    # 1. Intersect with flat entry face (z = z_flat)
    t1 = (z_flat - origin[2]) / (direction[2] + 1e-30)
    hit1 = origin + t1 * direction

    # Soft aperture mask at entry
    r_hit = jnp.sqrt(hit1[0] ** 2 + hit1[1] ** 2)
    aperture_weight = jax.nn.sigmoid((aperture_radius - r_hit) * 1e3)

    # 2. Refract at flat face (air -> glass)
    # Convention: N must point toward SOURCE medium for cosi = -dot(I,N) > 0.
    # Flat face outward normal is [0,0,-1] (pointing toward incoming beam = source).
    flat_normal = jnp.array([0.0, 0.0, -1.0])
    dir_glass, w1 = _snell_refract(direction, flat_normal, 1.0, n_glass)

    # 3. Intersect with aspheric exit surface
    t2, hit2 = _ray_aspheric_intersect(hit1, dir_glass, c, k, alphas, z_asph)

    # 4. Refract at aspheric face (glass -> air)
    # Outward normal points +Z (away from glass). When exiting, N toward source
    # (glass interior) = -outward_normal.
    normal_out = _surface_normal_aspheric(hit2, c, k, alphas, z_asph)
    dir_air, w2 = _snell_refract(dir_glass, -normal_out, n_glass, 1.0)

    # 5. Propagate to target plane
    t3 = (target_z - hit2[2]) / (dir_air[2] + 1e-30)
    target_point = hit2 + t3 * dir_air

    weight = aperture_weight * w1 * w2

    # Input radius at the lens entry face
    input_r = jnp.sqrt(hit1[0] ** 2 + hit1[1] ** 2)

    return target_point[:2], weight, input_r


# Vectorized over a ray bundle
_trace_ray_vmap = jax.vmap(trace_ray, in_axes=(0, 0, None, None, None, None, None, None, None))


@partial(jax.jit, static_argnums=())
def trace_bundle(origins, directions, c, k, alphas, center_thickness, n_glass, aperture_radius, target_z):
    """
    Trace a bundle of rays through a plano-aspheric lens.

    Args:
        origins: (N, 3) ray origin positions
        directions: (N, 3) ray direction vectors
        c: curvature (scalar)
        k: conic constant (scalar)
        alphas: (M,) polynomial coefficients
        center_thickness: lens thickness (scalar)
        n_glass: refractive index (scalar)
        aperture_radius: lens aperture (scalar)
        target_z: target plane z-position (scalar)

    Returns:
        positions: (N, 2) x,y positions at target plane
        weights: (N,) intensity weights
        input_radii: (N,) radial positions at lens entry face
    """
    positions, weights, input_radii = _trace_ray_vmap(
        origins, directions, c, k, alphas, center_thickness, n_glass, aperture_radius, target_z
    )
    return positions, weights, input_radii


# ---------------------------------------------------------------------------
# Gaussian beam ray generation (JAX-compatible)
# ---------------------------------------------------------------------------

def gaussian_beam_rays_jax(n_rays, waist_radius, wavelength, waist_z, launch_z, key):
    """
    Generate Gaussian beam rays as JAX arrays.

    Replicates the logic from mesh-optics-master/ray_sources.py:gaussian_beam_rays
    but returns pure JAX arrays for differentiable tracing.

    Args:
        n_rays: number of rays
        waist_radius: w0 (1/e^2 intensity radius at waist)
        wavelength: in meters
        waist_z: z-position of beam waist
        launch_z: z-position of launch plane
        key: JAX PRNG key

    Returns:
        origins: (n_rays, 3) array
        directions: (n_rays, 3) array
    """
    w0 = waist_radius
    zR = jnp.pi * w0 * w0 / wavelength
    z = launch_z - waist_z
    w_z = w0 * jnp.sqrt(1.0 + (z / zR) ** 2)

    # Wavefront radius
    Rz = jnp.where(jnp.abs(z) < 1e-15, 1e30, z * (1.0 + (zR / z) ** 2))

    # Sample from intensity distribution: I(r) = exp(-2r^2/w^2)
    # In 2D: f(x,y) ∝ exp(-2(x²+y²)/w²), so x,y ~ N(0, sigma²) with sigma = w/2
    sigma = w_z / 2.0
    key_x, key_y = jax.random.split(key)
    xs = jax.random.normal(key_x, (n_rays,)) * sigma
    ys = jax.random.normal(key_y, (n_rays,)) * sigma

    # Origins at launch plane
    origins = jnp.stack([xs, ys, jnp.full(n_rays, launch_z)], axis=-1)

    # Directions follow wavefront curvature
    # Wavefront center is at z = launch_z + Rz
    wf_center = jnp.array([0.0, 0.0, launch_z + Rz])
    dirs = wf_center[None, :] - origins  # (N, 3)
    dirs = dirs / (jnp.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-30)

    return origins, dirs


# ---------------------------------------------------------------------------
# Ray path extraction for visualization
# ---------------------------------------------------------------------------

def extract_ray_paths(origins, directions, c, k, alphas, center_thickness, n_glass, aperture_radius, target_z):
    """
    Trace rays and return full paths for visualization.

    Returns (N, 4, 3) array: origin -> flat hit -> aspheric hit -> target.
    """
    z_flat = -center_thickness * 0.5
    z_asph = center_thickness * 0.5

    def _trace_path(origin, direction):
        # Flat entry hit
        t1 = (z_flat - origin[2]) / (direction[2] + 1e-30)
        hit1 = origin + t1 * direction

        # Refract at flat face (air -> glass)
        flat_normal = jnp.array([0.0, 0.0, -1.0])
        dir_glass, _ = _snell_refract(direction, flat_normal, 1.0, n_glass)

        # Aspheric exit hit
        _, hit2 = _ray_aspheric_intersect(hit1, dir_glass, c, k, alphas, z_asph)

        # Refract at aspheric face (glass -> air)
        normal_out = _surface_normal_aspheric(hit2, c, k, alphas, z_asph)
        dir_air, _ = _snell_refract(dir_glass, -normal_out, n_glass, 1.0)

        # Target
        t3 = (target_z - hit2[2]) / (dir_air[2] + 1e-30)
        target = hit2 + t3 * dir_air

        return jnp.stack([origin, hit1, hit2, target])

    paths = jax.vmap(_trace_path)(origins, directions)
    return paths  # (N, 4, 3)
