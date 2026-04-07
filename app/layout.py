"""Dash app layout definition."""

from dash import html, dcc


def slider_group(label, id, min_val, max_val, value, step=None, marks=None):
    """Helper to create a labeled slider."""
    if step is None:
        step = (max_val - min_val) / 100
    return html.Div([
        html.Label(label, style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '2px'}),
        dcc.Slider(
            id=id, min=min_val, max=max_val, value=value, step=step,
            marks=marks,
            tooltip={'placement': 'bottom', 'always_visible': False},
        ),
    ], style={'marginBottom': '12px'})


def input_group(label, id, value, type='number', step=None):
    """Helper to create a labeled input."""
    return html.Div([
        html.Label(label, style={'fontWeight': '600', 'fontSize': '13px', 'marginRight': '8px'}),
        dcc.Input(id=id, value=value, type=type, step=step,
                  style={'width': '100px', 'padding': '4px 8px', 'border': '1px solid #555',
                         'borderRadius': '4px', 'background': '#2a2a2a', 'color': '#eee'}),
    ], style={'marginBottom': '10px', 'display': 'flex', 'alignItems': 'center'})


def make_layout():
    return html.Div([
        # Header
        html.Div([
            html.H1('Beam Shaper Optimizer',
                     style={'margin': '0', 'fontSize': '22px', 'fontWeight': '700'}),
            html.P('Gaussian to flat-top via aspheric lens optimization',
                    style={'margin': '2px 0 0 0', 'fontSize': '13px', 'color': '#aaa'}),
        ], style={'padding': '12px 20px', 'borderBottom': '1px solid #333'}),

        # Main content
        html.Div([
            # Left panel - controls
            html.Div([
                # View toggle
                html.Div([
                    html.Label('View', style={'fontWeight': '700', 'marginRight': '12px'}),
                    dcc.RadioItems(
                        id='view-toggle',
                        options=[
                            {'label': ' 3D', 'value': '3d'},
                            {'label': ' 2D Cross-section', 'value': '2d'},
                        ],
                        value='3d',
                        inline=True,
                        style={'display': 'inline-flex', 'gap': '16px'},
                        inputStyle={'marginRight': '4px'},
                    ),
                ], style={'marginBottom': '16px', 'padding': '8px 0', 'borderBottom': '1px solid #333'}),

                # Lens parameters
                html.H3('Lens Parameters', style={'fontSize': '15px', 'margin': '0 0 10px 0', 'color': '#8bb4f0'}),
                slider_group('Curvature (1/R) [1/m]', 'curvature-slider', 0.5, 20.0, 6.25, step=0.1),
                slider_group('Conic constant k', 'conic-slider', -5.0, 5.0, 0.0, step=0.05),
                slider_group('Alpha 4 (r^4 coeff)', 'alpha4-slider', -500, 500, 0.0, step=1.0),
                slider_group('Alpha 6 (r^6 coeff)', 'alpha6-slider', -5e4, 5e4, 0.0, step=100.0),
                slider_group('Alpha 8 (r^8 coeff)', 'alpha8-slider', -5e6, 5e6, 0.0, step=1e4),
                slider_group('Alpha 10 (r^10 coeff)', 'alpha10-slider', -5e8, 5e8, 0.0, step=1e6),

                html.Hr(style={'borderColor': '#333', 'margin': '16px 0'}),

                # Beam parameters
                html.H3('Beam Parameters', style={'fontSize': '15px', 'margin': '0 0 10px 0', 'color': '#8bb4f0'}),
                input_group('Waist w0 (mm)', 'waist-input', 40, step=1),
                input_group('N rays', 'nrays-input', 500, step=50),

                html.Hr(style={'borderColor': '#333', 'margin': '16px 0'}),

                # Target parameters
                html.H3('Target', style={'fontSize': '15px', 'margin': '0 0 10px 0', 'color': '#8bb4f0'}),
                slider_group('Target z (mm)', 'target-z-slider', 50, 500, 200, step=5),
                slider_group('Target radius (mm)', 'target-r-slider', 1, 50, 20, step=0.5),

                html.Hr(style={'borderColor': '#333', 'margin': '16px 0'}),

                # Optimization controls
                html.H3('Optimization', style={'fontSize': '15px', 'margin': '0 0 10px 0', 'color': '#8bb4f0'}),
                input_group('Learning rate', 'lr-input', 0.01, step=0.001),
                input_group('Max iterations', 'maxiter-input', 300, step=50),
                html.Div([
                    html.Button('Analytical Design', id='analytical-btn', n_clicks=0,
                                style={'padding': '8px 16px', 'background': '#4CAF50',
                                       'color': 'white', 'border': 'none', 'borderRadius': '6px',
                                       'cursor': 'pointer', 'fontWeight': '600', 'marginRight': '8px'}),
                    html.Button('Gradient Optimize', id='run-btn', n_clicks=0,
                                style={'padding': '8px 16px', 'background': '#2196F3',
                                       'color': 'white', 'border': 'none', 'borderRadius': '6px',
                                       'cursor': 'pointer', 'fontWeight': '600', 'marginRight': '8px'}),
                    html.Button('Stop', id='stop-btn', n_clicks=0,
                                style={'padding': '8px 12px', 'background': '#f44336',
                                       'color': 'white', 'border': 'none', 'borderRadius': '6px',
                                       'cursor': 'pointer', 'fontWeight': '600', 'marginRight': '8px'}),
                    html.Button('Reset', id='reset-btn', n_clicks=0,
                                style={'padding': '8px 12px', 'background': '#555',
                                       'color': 'white', 'border': 'none', 'borderRadius': '6px',
                                       'cursor': 'pointer', 'fontWeight': '600'}),
                ], style={'marginBottom': '10px', 'display': 'flex', 'flexWrap': 'wrap', 'gap': '4px'}),
                html.Div(id='opt-status', style={'fontSize': '13px', 'color': '#aaa', 'minHeight': '20px'}),

            ], style={
                'width': '320px', 'minWidth': '320px', 'padding': '16px',
                'overflowY': 'auto', 'borderRight': '1px solid #333',
            }),

            # Right panel - visualizations
            html.Div([
                # 3D View
                html.Div(id='view-3d-container', children=[
                    dcc.Graph(id='graph-3d', style={'height': '500px'}),
                ]),

                # 2D Views
                html.Div(id='view-2d-container', children=[
                    html.Div([
                        html.Div([
                            dcc.Graph(id='graph-cross-section'),
                        ], style={'flex': '1'}),
                        html.Div([
                            dcc.Graph(id='graph-lens-profile'),
                        ], style={'flex': '1'}),
                    ], style={'display': 'flex', 'gap': '8px'}),
                ], style={'display': 'none'}),

                # Always visible: radial profile + loss
                html.Div([
                    html.Div([
                        dcc.Graph(id='graph-radial-profile'),
                    ], style={'flex': '2'}),
                    html.Div([
                        dcc.Graph(id='graph-loss'),
                    ], style={'flex': '1'}),
                ], style={'display': 'flex', 'gap': '8px'}),

            ], style={'flex': '1', 'padding': '8px', 'overflowY': 'auto'}),

        ], style={'display': 'flex', 'height': 'calc(100vh - 60px)'}),

        # Stores
        dcc.Store(id='opt-state', data={'running': False, 'loss_history': [], 'iteration': 0}),
        dcc.Interval(id='opt-interval', interval=500, disabled=True),

    ], style={
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        'background': '#1a1a1a', 'color': '#eee', 'height': '100vh', 'overflow': 'hidden',
    })
