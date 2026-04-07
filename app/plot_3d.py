"""3D visualization of the optical scene using Plotly."""

import numpy as np
import plotly.graph_objects as go


def tris_to_mesh3d(tris, color='lightblue', opacity=0.4, name='Lens'):
    """
    Convert (N, 3, 3) triangle array to a Plotly Mesh3d trace.

    Deduplicates vertices and builds face index arrays.
    """
    tris = np.asarray(tris)
    n_tris = tris.shape[0]

    # Flatten all vertices
    all_verts = tris.reshape(-1, 3)  # (3N, 3)

    # Deduplicate via rounding
    rounded = np.round(all_verts, decimals=10)
    _, unique_idx, inverse = np.unique(
        rounded, axis=0, return_index=True, return_inverse=True
    )

    verts = all_verts[unique_idx]

    # Face indices
    face_indices = inverse.reshape(n_tris, 3)

    return go.Mesh3d(
        x=verts[:, 0] * 1e3,  # Convert to mm for display
        y=verts[:, 1] * 1e3,
        z=verts[:, 2] * 1e3,
        i=face_indices[:, 0],
        j=face_indices[:, 1],
        k=face_indices[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=True,
        showlegend=True,
    )


def ray_paths_to_scatter3d(paths, max_rays=100, color='crimson', width=1.5, opacity=0.6):
    """
    Convert ray paths (N, M, 3) to Plotly Scatter3d lines.

    paths: array of shape (N, num_points, 3)
    Inserts None between rays so they render as separate lines.
    """
    paths = np.asarray(paths)
    n_rays = min(paths.shape[0], max_rays)

    xs, ys, zs = [], [], []
    for i in range(n_rays):
        for j in range(paths.shape[1]):
            xs.append(paths[i, j, 0] * 1e3)
            ys.append(paths[i, j, 1] * 1e3)
            zs.append(paths[i, j, 2] * 1e3)
        xs.append(None)
        ys.append(None)
        zs.append(None)

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color=color, width=width),
        opacity=opacity,
        name='Rays',
        showlegend=True,
    )


def target_plane_disk(target_z, radius, color='rgba(100,100,255,0.15)', n_points=64):
    """Create a semi-transparent disk at the target plane."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    r = np.array([0, radius])
    theta_grid, r_grid = np.meshgrid(theta, r)
    x = r_grid * np.cos(theta_grid) * 1e3
    y = r_grid * np.sin(theta_grid) * 1e3
    z = np.full_like(x, target_z * 1e3)

    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name='Target plane',
        showlegend=True,
        opacity=0.3,
    )


def make_3d_figure(lens_tris=None, ray_paths=None, target_z=None, aperture_radius=None):
    """
    Create the full 3D scene figure.

    Args:
        lens_tris: (N, 3, 3) triangle array or None
        ray_paths: (N, M, 3) ray path array or None
        target_z: z-position of target plane
        aperture_radius: lens aperture radius for target plane size
    """
    traces = []

    if lens_tris is not None:
        traces.append(tris_to_mesh3d(lens_tris))

    if ray_paths is not None:
        traces.append(ray_paths_to_scatter3d(ray_paths))

    if target_z is not None:
        r = aperture_radius if aperture_radius else 0.05
        traces.append(target_plane_disk(target_z, r * 1.5))

    fig = go.Figure(data=traces)

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        legend=dict(x=0.02, y=0.98),
    )

    return fig
