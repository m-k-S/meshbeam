"""2D cross-section and profile plots using Plotly."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_cross_section_figure(positions, weights, target_radius=None):
    """
    2D scatter plot of ray positions at the target plane.

    Args:
        positions: (N, 2) array of (x, y) positions
        weights: (N,) intensity weights
        target_radius: if set, draw target circle
    """
    positions = np.asarray(positions)
    weights = np.asarray(weights)

    fig = go.Figure()

    # Ray scatter
    fig.add_trace(go.Scatter(
        x=positions[:, 0] * 1e3,
        y=positions[:, 1] * 1e3,
        mode='markers',
        marker=dict(
            size=4,
            color=weights,
            colorscale='Inferno',
            showscale=True,
            colorbar=dict(title='Weight', thickness=15),
            cmin=0, cmax=1,
        ),
        name='Rays',
    ))

    # Target circle
    if target_radius is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=target_radius * np.cos(theta) * 1e3,
            y=target_radius * np.sin(theta) * 1e3,
            mode='lines',
            line=dict(color='cyan', width=2, dash='dash'),
            name=f'Target r={target_radius*1e3:.1f}mm',
        ))

    fig.update_layout(
        xaxis_title='X (mm)',
        yaxis_title='Y (mm)',
        height=400,
        margin=dict(l=50, r=10, t=30, b=40),
        xaxis=dict(scaleanchor='y'),
    )

    return fig


def make_radial_profile_figure(positions, weights, target_radius, n_eval=100, bandwidth=None):
    """
    Radial intensity profile: KDE estimate vs flat-top target.

    Args:
        positions: (N, 2) ray positions
        weights: (N,) intensity weights
        target_radius: desired flat-top radius
        n_eval: number of evaluation points
        bandwidth: KDE bandwidth
    """
    positions = np.asarray(positions)
    weights = np.asarray(weights)

    r_samples = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    r_eval = np.linspace(0, target_radius * 2, n_eval)

    if bandwidth is None:
        bandwidth = target_radius / n_eval * 3.0

    # KDE
    diff = r_eval[:, None] - r_samples[None, :]
    kernels = np.exp(-0.5 * (diff / bandwidth) ** 2)
    density = np.sum(kernels * weights[None, :], axis=1)
    if np.max(density) > 0:
        density = density / np.max(density)

    # Target: ideal flat-top with soft edge
    from scipy.special import expit
    target = expit((target_radius - r_eval) * 200.0 / target_radius)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=r_eval * 1e3,
        y=density,
        mode='lines',
        line=dict(color='#ff6b35', width=2.5),
        name='Actual profile',
    ))

    fig.add_trace(go.Scatter(
        x=r_eval * 1e3,
        y=target,
        mode='lines',
        line=dict(color='cyan', width=2, dash='dash'),
        name='Target (flat-top)',
    ))

    fig.update_layout(
        xaxis_title='Radius (mm)',
        yaxis_title='Normalized intensity',
        yaxis=dict(range=[-0.05, 1.15]),
        height=300,
        margin=dict(l=50, r=10, t=30, b=40),
        legend=dict(x=0.6, y=0.95),
    )

    return fig


def make_loss_figure(loss_history):
    """Plot optimization loss vs iteration."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(loss_history))),
        y=loss_history,
        mode='lines',
        line=dict(color='#ff6b35', width=2),
        name='Loss',
    ))

    fig.update_layout(
        xaxis_title='Iteration',
        yaxis_title='Loss',
        height=250,
        margin=dict(l=50, r=10, t=30, b=40),
    )

    return fig


def make_lens_profile_figure(profile, center_thickness):
    """
    2D cross-section of the lens profile (r vs z).

    Args:
        profile: AsphericProfile object
        center_thickness: lens center thickness
    """
    from beamshaper.aspheric import aspheric_sag

    r = np.linspace(0, profile.aperture_radius, 200)
    z_vertex = center_thickness * 0.5
    z_plane = -center_thickness * 0.5

    sag_vals = aspheric_sag(r, profile)
    z_asph = z_vertex - sag_vals

    fig = go.Figure()

    # Aspheric surface (both sides for symmetry)
    fig.add_trace(go.Scatter(
        x=np.concatenate([-r[::-1], r]) * 1e3,
        y=np.concatenate([z_asph[::-1], z_asph]) * 1e3,
        mode='lines',
        line=dict(color='#2196F3', width=2.5),
        name='Aspheric face',
    ))

    # Flat face
    fig.add_trace(go.Scatter(
        x=np.array([-profile.aperture_radius, profile.aperture_radius]) * 1e3,
        y=np.array([z_plane, z_plane]) * 1e3,
        mode='lines',
        line=dict(color='#4CAF50', width=2.5),
        name='Flat face',
    ))

    # Rim
    fig.add_trace(go.Scatter(
        x=np.array([profile.aperture_radius, profile.aperture_radius]) * 1e3,
        y=np.array([z_plane, float(z_asph[-1])]) * 1e3,
        mode='lines',
        line=dict(color='gray', width=1.5),
        name='Rim',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=np.array([-profile.aperture_radius, -profile.aperture_radius]) * 1e3,
        y=np.array([z_plane, float(z_asph[-1])]) * 1e3,
        mode='lines',
        line=dict(color='gray', width=1.5),
        showlegend=False,
    ))

    fig.update_layout(
        xaxis_title='r (mm)',
        yaxis_title='z (mm)',
        height=300,
        margin=dict(l=50, r=10, t=30, b=40),
        yaxis=dict(scaleanchor='x'),
    )

    return fig
