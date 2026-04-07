#!/usr/bin/env python3
"""Launch the beam shaper optimizer web UI."""

from app.main import create_app

if __name__ == '__main__':
    app = create_app()
    print("Starting Beam Shaper Optimizer at http://localhost:8050")
    app.run(debug=True, port=8050)
