"""Dash callbacks for interactivity."""

import threading
import numpy as np
import jax
import jax.numpy as jnp
from dash import Input, Output, State, callback, no_update, ctx
import plotly.graph_objects as go

from beamshaper.aspheric import AsphericProfile, plano_aspheric_tris, aspheric_sag
from beamshaper.jax_tracer import trace_bundle, gaussian_beam_rays_jax, extract_ray_paths
from beamshaper.optimizer import optimize, BeamShapingConfig
from beamshaper.analytical import design_beam_shaper
from beamshaper.profiles import flat_top_target
from app.plot_3d import make_3d_figure
from app.plot_2d import (
    make_cross_section_figure, make_radial_profile_figure,
    make_loss_figure, make_lens_profile_figure,
)

# Shared state for optimization thread
_opt_lock = threading.Lock()
_opt_state = {
    'running': False,
    'stop': False,
    'iteration': 0,
    'loss_history': [],
    'latest_positions': None,
    'latest_weights': None,
    'latest_params': None,
    'done': False,
}

# Fixed parameters
CENTER_THICKNESS = 50e-3
N_GLASS = 1.52
APERTURE_RADIUS = 50e-3
WAVELENGTH = 780e-9
WAIST_Z = -0.10
LAUNCH_Z = -0.06


def _empty_fig(height=300):
    fig = go.Figure()
    fig.update_layout(
        height=height, margin=dict(l=50, r=10, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#aaa',
    )
    return fig


def _style_fig(fig):
    """Apply dark theme to a plotly figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,26,0.8)',
        font_color='#ccc',
        xaxis=dict(gridcolor='#333', zerolinecolor='#444'),
        yaxis=dict(gridcolor='#333', zerolinecolor='#444'),
    )
    return fig


def _trace_with_params(curvature, conic, alphas_list, waist_mm, n_rays, target_z_mm):
    """Run the JAX tracer with given parameters. Returns (positions, weights, paths)."""
    c = float(curvature)
    k = float(conic)
    alphas = jnp.array(alphas_list, dtype=float)
    target_z = target_z_mm * 1e-3
    waist = waist_mm * 1e-3

    key = jax.random.PRNGKey(42)
    origins, directions = gaussian_beam_rays_jax(
        int(n_rays), waist, WAVELENGTH, WAIST_Z, LAUNCH_Z, key
    )

    positions, weights, _input_radii = trace_bundle(
        origins, directions, c, k, alphas,
        CENTER_THICKNESS, N_GLASS, APERTURE_RADIUS, target_z,
    )

    paths = extract_ray_paths(
        origins, directions, c, k, alphas,
        CENTER_THICKNESS, N_GLASS, APERTURE_RADIUS, target_z,
    )

    return (
        np.asarray(positions), np.asarray(weights), np.asarray(paths),
        c, k, alphas_list,
    )


def register_callbacks(app):
    """Register all Dash callbacks."""

    # ----- View toggle -----
    @app.callback(
        Output('view-3d-container', 'style'),
        Output('view-2d-container', 'style'),
        Input('view-toggle', 'value'),
    )
    def toggle_view(view):
        if view == '3d':
            return {'display': 'block'}, {'display': 'none'}
        return {'display': 'none'}, {'display': 'flex', 'flexDirection': 'column'}

    # ----- Main update: retrace rays when parameters change -----
    @app.callback(
        Output('graph-3d', 'figure'),
        Output('graph-cross-section', 'figure'),
        Output('graph-radial-profile', 'figure'),
        Output('graph-lens-profile', 'figure'),
        Input('curvature-slider', 'value'),
        Input('conic-slider', 'value'),
        Input('alpha4-slider', 'value'),
        Input('alpha6-slider', 'value'),
        Input('alpha8-slider', 'value'),
        Input('alpha10-slider', 'value'),
        Input('waist-input', 'value'),
        Input('nrays-input', 'value'),
        Input('target-z-slider', 'value'),
        Input('target-r-slider', 'value'),
    )
    def update_plots(curvature, conic, a4, a6, a8, a10, waist_mm, n_rays, target_z_mm, target_r_mm):
        if any(v is None for v in [curvature, conic, a4, a6, a8, a10, waist_mm, n_rays, target_z_mm, target_r_mm]):
            return _empty_fig(500), _empty_fig(400), _empty_fig(300), _empty_fig(300)

        alphas_list = [float(a4), float(a6), float(a8), float(a10)]
        target_r = target_r_mm * 1e-3

        try:
            positions, weights, paths, c, k, alphas = _trace_with_params(
                curvature, conic, alphas_list, waist_mm, n_rays, target_z_mm
            )
        except Exception as e:
            print(f"Trace error: {e}")
            return _empty_fig(500), _empty_fig(400), _empty_fig(300), _empty_fig(300)

        # Generate lens mesh for 3D view
        profile = AsphericProfile(
            curvature=c, conic_constant=k,
            alphas=tuple(alphas_list), aperture_radius=APERTURE_RADIUS,
        )
        try:
            tris = plano_aspheric_tris(
                profile, CENTER_THICKNESS,
                radial_segments=24, azimuth_segments=64,
            )
        except ValueError:
            tris = None

        # 3D figure
        fig_3d = make_3d_figure(
            lens_tris=tris, ray_paths=paths,
            target_z=target_z_mm * 1e-3, aperture_radius=APERTURE_RADIUS,
        )
        fig_3d.update_layout(
            scene=dict(
                bgcolor='rgba(26,26,26,1)',
                xaxis=dict(backgroundcolor='rgba(26,26,26,1)', gridcolor='#333', color='#aaa'),
                yaxis=dict(backgroundcolor='rgba(26,26,26,1)', gridcolor='#333', color='#aaa'),
                zaxis=dict(backgroundcolor='rgba(26,26,26,1)', gridcolor='#333', color='#aaa'),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccc',
        )

        # 2D cross section
        fig_cross = make_cross_section_figure(positions, weights, target_radius=target_r)
        _style_fig(fig_cross)

        # Radial profile
        fig_radial = make_radial_profile_figure(positions, weights, target_r)
        _style_fig(fig_radial)

        # Lens profile
        fig_lens = make_lens_profile_figure(profile, CENTER_THICKNESS)
        _style_fig(fig_lens)

        return fig_3d, fig_cross, fig_radial, fig_lens

    # ----- Analytical design -----
    @app.callback(
        Output('curvature-slider', 'value', allow_duplicate=True),
        Output('conic-slider', 'value', allow_duplicate=True),
        Output('alpha4-slider', 'value', allow_duplicate=True),
        Output('alpha6-slider', 'value', allow_duplicate=True),
        Output('alpha8-slider', 'value', allow_duplicate=True),
        Output('alpha10-slider', 'value', allow_duplicate=True),
        Output('opt-status', 'children', allow_duplicate=True),
        Input('analytical-btn', 'n_clicks'),
        State('waist-input', 'value'),
        State('target-z-slider', 'value'),
        State('target-r-slider', 'value'),
        prevent_initial_call=True,
    )
    def run_analytical_design(n_clicks, waist_mm, target_z_mm, target_r_mm):
        if n_clicks == 0:
            return (no_update,) * 7

        try:
            result = design_beam_shaper(
                w0=(waist_mm or 40) * 1e-3,
                target_radius=(target_r_mm or 20) * 1e-3,
                target_z=(target_z_mm or 200) * 1e-3,
                n_glass=N_GLASS,
                center_thickness=CENTER_THICKNESS,
                aperture_radius=APERTURE_RADIUS,
                n_alpha_coeffs=4,
            )
            c = result['curvature']
            k = result['conic']
            alphas = result['alphas']
            status = f'Analytical design: c={c:.4f}, k={k:.4f}'
            return (
                round(c, 4), round(k, 4),
                round(float(alphas[0]), 2), round(float(alphas[1]), 2),
                round(float(alphas[2]), 2), round(float(alphas[3]), 2),
                status,
            )
        except Exception as e:
            return (no_update,) * 6 + (f'Error: {e}',)

    # ----- Optimization: run in background thread -----
    @app.callback(
        Output('opt-interval', 'disabled', allow_duplicate=True),
        Output('opt-status', 'children', allow_duplicate=True),
        Input('run-btn', 'n_clicks'),
        State('curvature-slider', 'value'),
        State('conic-slider', 'value'),
        State('alpha4-slider', 'value'),
        State('alpha6-slider', 'value'),
        State('alpha8-slider', 'value'),
        State('alpha10-slider', 'value'),
        State('waist-input', 'value'),
        State('nrays-input', 'value'),
        State('target-z-slider', 'value'),
        State('target-r-slider', 'value'),
        State('lr-input', 'value'),
        State('maxiter-input', 'value'),
        prevent_initial_call=True,
    )
    def start_optimization(n_clicks, curvature, conic, a4, a6, a8, a10,
                           waist_mm, n_rays, target_z_mm, target_r_mm, lr, maxiter):
        if n_clicks == 0:
            return True, ''

        with _opt_lock:
            if _opt_state['running']:
                return True, 'Already running...'
            _opt_state['running'] = True
            _opt_state['stop'] = False
            _opt_state['iteration'] = 0
            _opt_state['loss_history'] = []
            _opt_state['done'] = False

        config = BeamShapingConfig(
            n_rays=int(n_rays or 500),
            waist_radius=(waist_mm or 40) * 1e-3,
            wavelength=WAVELENGTH,
            waist_z=WAIST_Z,
            launch_z=LAUNCH_Z,
            curvature=float(curvature or 6.25),
            conic_constant=float(conic or 0.0),
            n_alpha_coeffs=4,
            center_thickness=CENTER_THICKNESS,
            n_glass=N_GLASS,
            aperture_radius=APERTURE_RADIUS,
            target_z=(target_z_mm or -200) * 1e-3,
            target_radius=(target_r_mm or 20) * 1e-3,
            learning_rate=float(lr or 0.01),
            max_iterations=int(maxiter or 300),
        )

        def run_opt():
            def cb(iteration, loss, params, positions, weights):
                with _opt_lock:
                    _opt_state['iteration'] = iteration
                    _opt_state['loss_history'].append(float(loss))
                    _opt_state['latest_positions'] = np.asarray(positions)
                    _opt_state['latest_weights'] = np.asarray(weights)
                    _opt_state['latest_params'] = {
                        'curvature': float(params['curvature']),
                        'conic': float(params['conic']),
                        'alphas': [float(a) for a in params['alphas']],
                    }

            def stop_check():
                with _opt_lock:
                    return _opt_state['stop']

            try:
                optimize(config, callback=cb, stop_flag=stop_check)
            finally:
                with _opt_lock:
                    _opt_state['running'] = False
                    _opt_state['done'] = True

        thread = threading.Thread(target=run_opt, daemon=True)
        thread.start()

        return False, 'Optimizing...'

    # ----- Stop optimization -----
    @app.callback(
        Output('opt-status', 'children', allow_duplicate=True),
        Input('stop-btn', 'n_clicks'),
        prevent_initial_call=True,
    )
    def stop_optimization(n_clicks):
        if n_clicks > 0:
            with _opt_lock:
                _opt_state['stop'] = True
            return 'Stopping...'
        return no_update

    # ----- Reset sliders -----
    @app.callback(
        Output('curvature-slider', 'value'),
        Output('conic-slider', 'value'),
        Output('alpha4-slider', 'value'),
        Output('alpha6-slider', 'value'),
        Output('alpha8-slider', 'value'),
        Output('alpha10-slider', 'value'),
        Input('reset-btn', 'n_clicks'),
        prevent_initial_call=True,
    )
    def reset_params(n_clicks):
        if n_clicks > 0:
            return 6.25, 0.0, 0.0, 0.0, 0.0, 0.0
        return no_update, no_update, no_update, no_update, no_update, no_update

    # ----- Poll optimization progress -----
    @app.callback(
        Output('graph-loss', 'figure'),
        Output('opt-status', 'children', allow_duplicate=True),
        Output('opt-interval', 'disabled', allow_duplicate=True),
        Output('curvature-slider', 'value', allow_duplicate=True),
        Output('conic-slider', 'value', allow_duplicate=True),
        Output('alpha4-slider', 'value', allow_duplicate=True),
        Output('alpha6-slider', 'value', allow_duplicate=True),
        Output('alpha8-slider', 'value', allow_duplicate=True),
        Output('alpha10-slider', 'value', allow_duplicate=True),
        Input('opt-interval', 'n_intervals'),
        prevent_initial_call=True,
    )
    def poll_optimization(n_intervals):
        with _opt_lock:
            loss_history = list(_opt_state['loss_history'])
            iteration = _opt_state['iteration']
            done = _opt_state['done']
            running = _opt_state['running']
            params = _opt_state.get('latest_params')

        fig_loss = make_loss_figure(loss_history) if loss_history else _empty_fig(250)
        _style_fig(fig_loss)

        if done and not running:
            status = f'Done! {len(loss_history)} iterations. Final loss: {loss_history[-1]:.6f}' if loss_history else 'Done!'
            # Update sliders to final params
            if params:
                return (fig_loss, status, True,
                        round(params['curvature'], 4),
                        round(params['conic'], 4),
                        round(params['alphas'][0], 2),
                        round(params['alphas'][1], 2),
                        round(params['alphas'][2], 2),
                        round(params['alphas'][3], 2))
            return fig_loss, status, True, no_update, no_update, no_update, no_update, no_update, no_update

        status = f'Iteration {iteration}/{len(loss_history)}...'
        if loss_history:
            status += f' Loss: {loss_history[-1]:.6f}'

        return fig_loss, status, False, no_update, no_update, no_update, no_update, no_update, no_update
