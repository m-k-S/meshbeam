"""
Full-system visualization: PM780 fiber → collimator → two-lens beam shaper → flat-top.

Generates a Plotly figure showing:
- 2D cross-section (r-z plane) with ray paths through the entire system
- Lens surface profiles
- Intensity profiles at key z-planes
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def trace_full_system(
    n_rays=80,
    w0_fiber=2.65e-6,
    wavelength=780e-9,
    f_collimator=11e-3,
    design_result=None,
    n_glass=1.52,
    target_z_beyond=100e-3,
    seed=42,
):
    """
    Trace rays from fiber tip through collimator and beam shaper.

    Returns dict with ray paths and system geometry.
    """
    rng = np.random.default_rng(seed)

    # Fiber parameters
    NA = wavelength / (np.pi * w0_fiber)  # paraxial divergence
    zR = np.pi * w0_fiber ** 2 / wavelength

    # System z-coordinates
    z_fiber = 0.0
    z_col = f_collimator                    # collimator at 1 focal length from fiber
    z_lens1_flat = z_col + 5e-3             # small gap after collimator
    t1 = design_result['t1']
    t2 = design_result['t2']
    sep = design_result['separation']
    z_lens1_asph = z_lens1_flat + t1
    z_lens2_asph = z_lens1_flat + sep - t2
    z_lens2_flat = z_lens1_flat + sep
    z_target = z_lens2_flat + target_z_beyond

    # Sag tables
    r1_tab = design_result['r']
    sag1_tab = design_result['sag1']
    r2_tab = design_result['R']
    sag2_tab = design_result['sag2']
    ap1 = r1_tab[-1]
    ap2 = r2_tab[-1]

    # Sample rays from fiber (intensity-proportional: sigma = w0/2)
    # But for visualization, use uniform spacing in angle for even ray distribution
    # Mix: some from Gaussian, some evenly spaced for visual clarity
    sigma_angle = NA / 2.0  # intensity-proportional half-angle sigma
    angles = rng.normal(0, sigma_angle, size=n_rays)
    angles = np.sort(angles)

    # Also add a few rays at specific angles for visual clarity
    extra = np.linspace(-NA * 0.95, NA * 0.95, 12)
    angles = np.concatenate([angles, extra])
    n_rays_total = len(angles)

    # For 2D cross-section, rays are in the r-z plane (y=0)
    paths = []
    for angle in angles:
        path = []

        # 1. Fiber tip
        r, z = 0.0, z_fiber
        path.append((r, z))

        # 2. Propagate to collimator (diverging)
        dr_dz = np.tan(angle)
        r_at_col = r + dr_dz * (z_col - z)
        path.append((r_at_col, z_col))

        # 3. Thin lens refraction at collimator
        # Thin lens: new angle = old_angle - r/f
        angle_after_col = angle - r_at_col / f_collimator
        dr_dz = np.tan(angle_after_col)

        # 4. Propagate to lens 1 flat face
        r_at_l1 = r_at_col + dr_dz * (z_lens1_flat - z_col)
        path.append((r_at_l1, z_lens1_flat))

        # Skip if outside aperture
        if abs(r_at_l1) > ap1 * 1.05:
            paths.append(np.array(path))
            continue

        # 5. Refract at lens 1 flat face (air → glass)
        sin_i = np.sin(np.arctan(dr_dz))
        sin_t = sin_i / n_glass
        angle_glass = np.arcsin(np.clip(sin_t, -1, 1))
        if dr_dz < 0:
            angle_glass = -abs(angle_glass)
        dr_dz_glass = np.tan(angle_glass)

        # 6. Propagate through lens 1 glass to aspheric face
        sag_at_r = np.interp(abs(r_at_l1), r1_tab, sag1_tab)
        z_asph_at_r = z_lens1_asph - sag_at_r  # z of aspheric surface at this r
        r_at_asph1 = r_at_l1 + dr_dz_glass * (z_asph_at_r - z_lens1_flat)
        path.append((r_at_asph1, z_asph_at_r))

        # 7. Refract at lens 1 aspheric (glass → air)
        dsag_dr = np.interp(abs(r_at_asph1), r1_tab, np.gradient(sag1_tab, r1_tab))
        if r_at_asph1 < 0:
            dsag_dr = -dsag_dr
        # Surface normal: (dsag/dr, 1)/norm → outward from glass (+Z at vertex)
        nx = dsag_dr
        nz = 1.0
        nn = np.sqrt(nx**2 + nz**2)
        nx, nz = nx/nn, nz/nn

        # Snell's law (vector form in 2D)
        di_r, di_z = np.sin(angle_glass), np.cos(angle_glass)
        if dr_dz_glass < 0:
            di_r = -abs(di_r)
        # Normal toward source (glass): -outward
        ns_r, ns_z = -nx, -nz
        cosi = -(di_r * ns_r + di_z * ns_z)
        eta = n_glass / 1.0
        k = 1 - eta**2 * (1 - cosi**2)
        if k < 0:
            paths.append(np.array(path))
            continue
        dt_r = eta * di_r + (eta * cosi - np.sqrt(k)) * ns_r
        dt_z = eta * di_z + (eta * cosi - np.sqrt(k)) * ns_z
        norm_t = np.sqrt(dt_r**2 + dt_z**2)
        dt_r, dt_z = dt_r/norm_t, dt_z/norm_t

        # 8. Propagate through air to lens 2 aspheric
        # Find where ray hits lens 2 aspheric surface
        # Approximate: propagate to z = z_lens2_asph (vertex), then refine
        if abs(dt_z) < 1e-10:
            paths.append(np.array(path))
            continue
        t_to_l2 = (z_lens2_asph - z_asph_at_r) / dt_z
        r_at_l2_approx = r_at_asph1 + dt_r * t_to_l2
        # Refine with sag
        sag2_at_r = np.interp(abs(r_at_l2_approx), r2_tab, sag2_tab)
        z_l2_at_r = z_lens2_asph - sag2_at_r
        t_to_l2 = (z_l2_at_r - z_asph_at_r) / dt_z
        r_at_asph2 = r_at_asph1 + dt_r * t_to_l2
        z_at_asph2 = z_asph_at_r + dt_z * t_to_l2
        path.append((r_at_asph2, z_at_asph2))

        # 9. Refract at lens 2 aspheric (air → glass)
        dsag2_dr = np.interp(abs(r_at_asph2), r2_tab, np.gradient(sag2_tab, r2_tab))
        if r_at_asph2 < 0:
            dsag2_dr = -dsag2_dr
        # Outward from glass at lens 2: -gradient (points toward lens 1, -Z)
        nx2, nz2 = -dsag2_dr, -1.0
        nn2 = np.sqrt(nx2**2 + nz2**2)
        nx2, nz2 = nx2/nn2, nz2/nn2
        # Normal toward source (air, coming from -Z side): same as outward from glass
        ns2_r, ns2_z = nx2, nz2
        cosi2 = -(dt_r * ns2_r + dt_z * ns2_z)
        eta2 = 1.0 / n_glass
        k2 = 1 - eta2**2 * (1 - cosi2**2)
        if k2 < 0:
            paths.append(np.array(path))
            continue
        dg2_r = eta2 * dt_r + (eta2 * cosi2 - np.sqrt(k2)) * ns2_r
        dg2_z = eta2 * dt_z + (eta2 * cosi2 - np.sqrt(k2)) * ns2_z
        norm_g2 = np.sqrt(dg2_r**2 + dg2_z**2)
        dg2_r, dg2_z = dg2_r/norm_g2, dg2_z/norm_g2

        # 10. Propagate through lens 2 glass to flat exit
        t_to_flat2 = (z_lens2_flat - z_at_asph2) / (dg2_z + 1e-30)
        r_at_flat2 = r_at_asph2 + dg2_r * t_to_flat2
        path.append((r_at_flat2, z_lens2_flat))

        # 11. Refract at lens 2 flat face (glass → air)
        sin_out = n_glass * dg2_r  # sin(angle) = n * sin(angle_glass)
        sin_out = np.clip(sin_out, -1, 1)
        angle_out = np.arcsin(sin_out)
        dr_dz_out = np.tan(angle_out)

        # 12. Propagate to target
        r_at_target = r_at_flat2 + dr_dz_out * (z_target - z_lens2_flat)
        path.append((r_at_target, z_target))

        paths.append(np.array(path))

    return {
        'paths': paths,
        'z_fiber': z_fiber,
        'z_col': z_col,
        'z_lens1_flat': z_lens1_flat,
        'z_lens1_asph': z_lens1_asph,
        'z_lens2_asph': z_lens2_asph,
        'z_lens2_flat': z_lens2_flat,
        'z_target': z_target,
        'ap1': ap1,
        'ap2': ap2,
        't1': t1,
        't2': t2,
        'r1_tab': r1_tab,
        'sag1_tab': sag1_tab,
        'r2_tab': r2_tab,
        'sag2_tab': sag2_tab,
    }


def make_system_figure(trace_result, title="PM780 Fiber → 15mm Flat-Top Beam Shaper"):
    """Create the full visualization figure."""
    tr = trace_result

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.08,
        subplot_titles=[title, "Radial intensity at target plane"],
    )

    # =========================================================
    # Top panel: 2D cross-section with rays and lens profiles
    # =========================================================

    # Ray paths (both +r and -r for symmetry)
    for path in tr['paths']:
        if len(path) < 2:
            continue
        r_vals = path[:, 0] * 1e3  # mm
        z_vals = path[:, 1] * 1e3  # mm
        fig.add_trace(go.Scatter(
            x=z_vals, y=r_vals,
            mode='lines', line=dict(color='rgba(255,80,40,0.25)', width=0.8),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)
        # Mirror
        fig.add_trace(go.Scatter(
            x=z_vals, y=-r_vals,
            mode='lines', line=dict(color='rgba(255,80,40,0.25)', width=0.8),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)

    # Fiber tip marker
    fig.add_trace(go.Scatter(
        x=[tr['z_fiber'] * 1e3], y=[0],
        mode='markers', marker=dict(color='#00ff88', size=8, symbol='diamond'),
        name='PM780 fiber tip',
    ), row=1, col=1)

    # Collimator (thin lens)
    z_col = tr['z_col'] * 1e3
    col_r = tr['ap1'] * 1.2 * 1e3
    fig.add_trace(go.Scatter(
        x=[z_col, z_col], y=[-col_r, col_r],
        mode='lines', line=dict(color='#aaddff', width=3),
        name='Collimator (f=11mm)',
    ), row=1, col=1)

    # Lens 1 profile (filled)
    r1 = tr['r1_tab']
    sag1 = tr['sag1_tab']
    z_flat1 = tr['z_lens1_flat'] * 1e3
    z_asph1 = tr['z_lens1_asph'] * 1e3

    # Aspheric surface of lens 1
    z_surf1 = (z_asph1 - sag1 * 1e3)
    r_mm = r1 * 1e3

    # Fill lens 1 body
    r_fill = np.concatenate([r_mm, r_mm[::-1]])
    z_fill = np.concatenate([np.full_like(r_mm, z_flat1), z_surf1[::-1]])
    fig.add_trace(go.Scatter(
        x=np.concatenate([z_fill, z_fill[::-1]]),
        y=np.concatenate([r_fill, -r_fill[::-1]]),
        fill='toself', fillcolor='rgba(100,180,255,0.3)',
        line=dict(color='#4488cc', width=1.5),
        name='Lens 1 (3mm)',
    ), row=1, col=1)

    # Lens 2 profile (filled)
    r2 = tr['r2_tab']
    sag2 = tr['sag2_tab']
    z_asph2 = tr['z_lens2_asph'] * 1e3
    z_flat2 = tr['z_lens2_flat'] * 1e3

    z_surf2 = (z_asph2 - sag2 * 1e3)
    r2_mm = r2 * 1e3

    r_fill2 = np.concatenate([r2_mm, r2_mm[::-1]])
    z_fill2 = np.concatenate([z_surf2, np.full_like(r2_mm, z_flat2)[::-1]])
    fig.add_trace(go.Scatter(
        x=np.concatenate([z_fill2, z_fill2[::-1]]),
        y=np.concatenate([r_fill2, -r_fill2[::-1]]),
        fill='toself', fillcolor='rgba(100,180,255,0.3)',
        line=dict(color='#4488cc', width=1.5),
        name='Lens 2 (15mm)',
    ), row=1, col=1)

    # Target plane
    z_tgt = tr['z_target'] * 1e3
    fig.add_trace(go.Scatter(
        x=[z_tgt, z_tgt], y=[-10, 10],
        mode='lines', line=dict(color='cyan', width=1.5, dash='dash'),
        name='Target plane',
    ), row=1, col=1)

    # Annotations
    fig.add_annotation(x=z_flat1, y=tr['ap1']*1.5*1e3, text="L1", showarrow=False,
                       font=dict(color='#88bbee', size=11), row=1, col=1)
    fig.add_annotation(x=(z_asph2+z_flat2)/2, y=tr['ap2']*1.1*1e3, text="L2", showarrow=False,
                       font=dict(color='#88bbee', size=11), row=1, col=1)
    fig.add_annotation(x=z_col, y=col_r*1.3, text="Collimator", showarrow=False,
                       font=dict(color='#aaddff', size=10), row=1, col=1)

    # =========================================================
    # Bottom panel: intensity profile at target
    # =========================================================
    target_r = []
    for path in tr['paths']:
        if len(path) >= 2:
            target_r.append(abs(path[-1, 0]))

    target_r = np.array(target_r) * 1e3  # mm
    target_r = target_r[target_r < 12]

    # Histogram as proxy for intensity
    bins = np.linspace(0, 10, 50)
    counts, edges = np.histogram(target_r, bins=bins)
    # Normalize by annular area
    centers = (edges[:-1] + edges[1:]) / 2
    areas = np.pi * (edges[1:]**2 - edges[:-1]**2)
    density = counts / (areas + 1e-10)
    if np.max(density) > 0:
        density = density / np.max(density)

    fig.add_trace(go.Scatter(
        x=centers, y=density,
        mode='lines', line=dict(color='#ff6b35', width=2.5),
        name='Intensity profile',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[7.5, 7.5], y=[0, 1.2],
        mode='lines', line=dict(color='cyan', width=1.5, dash='dash'),
        name='Target edge (7.5mm)',
    ), row=2, col=1)

    # Layout
    fig.update_layout(
        height=700, width=1100,
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1e1e2e',
        font=dict(color='#ccc'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)'),
        margin=dict(l=60, r=20, t=40, b=40),
    )

    fig.update_xaxes(title_text='z (mm)', gridcolor='#333', zerolinecolor='#444', row=1, col=1)
    fig.update_yaxes(title_text='r (mm)', gridcolor='#333', zerolinecolor='#444', row=1, col=1)
    fig.update_xaxes(title_text='r (mm)', gridcolor='#333', zerolinecolor='#444', row=2, col=1)
    fig.update_yaxes(title_text='Normalized intensity', gridcolor='#333', zerolinecolor='#444',
                     range=[-0.05, 1.3], row=2, col=1)

    return fig
