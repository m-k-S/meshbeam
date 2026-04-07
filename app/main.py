"""Dash application entry point."""

from dash import Dash
from app.layout import make_layout
from app.callbacks import register_callbacks


def create_app():
    app = Dash(
        __name__,
        assets_folder='assets',
        suppress_callback_exceptions=True,
    )
    app.title = 'Beam Shaper Optimizer'
    app.layout = make_layout()
    register_callbacks(app)
    return app
