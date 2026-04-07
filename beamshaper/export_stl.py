"""
STL export for 2PP-printed beam shaper lenses.

Generates watertight triangulated meshes from tabulated sag profiles.
Each lens is a solid of revolution: flat bottom face + aspheric top face + rim.

For 2PP printing:
- The flat face bonds to the glass substrate
- The aspheric surface is the optically active face
- The rim provides a clean edge and structural support
"""

import numpy as np
import struct


def _sag_surface_tris(r_table, sag_table, center_z, azimuth_segments=180,
                      outward_up=True):
    """
    Generate triangles for a surface of revolution from a sag table.

    The surface is z(r, theta) = center_z + sag_table(r) (interpolated).
    Triangulated on a polar grid using the r_table radial positions.

    Returns (N, 3, 3) triangle array.
    """
    n_radial = len(r_table)
    thetas = np.linspace(0, 2 * np.pi, azimuth_segments, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Generate surface points: (n_radial, azimuth_segments, 3)
    pts = np.zeros((n_radial, azimuth_segments, 3))
    for i, r in enumerate(r_table):
        z = center_z + sag_table[i]
        pts[i, :, 0] = r * cos_t
        pts[i, :, 1] = r * sin_t
        pts[i, :, 2] = z

    tris = []
    for j in range(n_radial - 1):
        for k in range(azimuth_segments):
            k2 = (k + 1) % azimuth_segments
            v00 = pts[j, k]
            v10 = pts[j + 1, k]
            v11 = pts[j + 1, k2]
            v01 = pts[j, k2]

            if outward_up:
                tris.append([v00, v10, v11])
                tris.append([v00, v11, v01])
            else:
                tris.append([v00, v11, v10])
                tris.append([v00, v01, v11])

    return np.array(tris, dtype=np.float32)


def _flat_disk_tris(radius, z, n_radial=20, azimuth_segments=180, outward_up=True):
    """Generate triangles for a flat circular disk at z=const."""
    r_table = np.linspace(0, radius, n_radial)
    sag_table = np.zeros(n_radial)
    return _sag_surface_tris(r_table, sag_table, z, azimuth_segments, outward_up)


def _rim_tris(r, z_bottom, z_top, azimuth_segments=180):
    """Generate triangles for a cylindrical rim connecting two z-levels."""
    thetas = np.linspace(0, 2 * np.pi, azimuth_segments, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    tris = []
    for k in range(azimuth_segments):
        k2 = (k + 1) % azimuth_segments
        # Bottom-left, bottom-right, top-right, top-left
        bl = np.array([r * cos_t[k], r * sin_t[k], z_bottom])
        br = np.array([r * cos_t[k2], r * sin_t[k2], z_bottom])
        tl = np.array([r * cos_t[k], r * sin_t[k], z_top])
        tr = np.array([r * cos_t[k2], r * sin_t[k2], z_top])
        # Outward-facing (radial direction)
        tris.append([bl, br, tr])
        tris.append([bl, tr, tl])

    return np.array(tris, dtype=np.float32)


def generate_lens_stl(r_table, sag_table, center_thickness, aperture_radius,
                      radial_segments=100, azimuth_segments=180):
    """
    Generate a watertight triangulated mesh for a plano-aspheric lens.

    The lens is oriented with:
    - Flat face at z = 0 (bonds to glass substrate)
    - Aspheric face at z = center_thickness - sag(r) (optically active)
    - sag(0) = 0 at the vertex, sag > 0 means surface drops below vertex

    Args:
        r_table: (N,) radial positions for the sag profile
        sag_table: (N,) sag values at each radius
        center_thickness: lens thickness at center (vertex)
        aperture_radius: clear aperture radius
        radial_segments: number of radial rings for the mesh
        azimuth_segments: number of angular segments

    Returns:
        (M, 3, 3) triangle array in meters
    """
    # Resample sag onto a regular grid for the mesh
    r_mesh = np.linspace(0, aperture_radius, radial_segments + 1)
    sag_mesh = np.interp(r_mesh, r_table, sag_table)

    # Top surface (aspheric): z = center_thickness - sag(r)
    # The surface drops below center_thickness as r increases
    aspheric_z = center_thickness - sag_mesh
    top_tris = _sag_surface_tris(
        r_mesh, aspheric_z, center_z=0.0,
        azimuth_segments=azimuth_segments, outward_up=True)

    # Bottom surface (flat): z = 0, normals pointing down
    bottom_tris = _flat_disk_tris(
        aperture_radius, z=0.0,
        n_radial=radial_segments + 1,
        azimuth_segments=azimuth_segments, outward_up=False)

    # Rim: connects aspheric edge to flat edge at r = aperture_radius
    z_top = float(aspheric_z[-1])  # z of aspheric surface at the edge
    z_bottom = 0.0
    rim_tris = _rim_tris(aperture_radius, z_bottom, z_top, azimuth_segments)

    return np.concatenate([top_tris, bottom_tris, rim_tris], axis=0)


def write_stl_binary(filepath, triangles, solid_name="lens"):
    """Write triangles to a binary STL file."""
    triangles = np.asarray(triangles, dtype=np.float32)
    n_tris = triangles.shape[0]

    with open(filepath, 'wb') as f:
        # 80-byte header
        header = f"Beam shaper lens: {solid_name}".encode('ascii')
        header = header[:80].ljust(80, b'\0')
        f.write(header)

        # Number of triangles
        f.write(struct.pack('<I', n_tris))

        for i in range(n_tris):
            v0, v1, v2 = triangles[i]
            # Compute normal
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal /= norm_len

            # Write: normal (3 floats) + 3 vertices (9 floats) + attribute (uint16)
            f.write(struct.pack('<fff', *normal))
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<H', 0))

    return filepath


def write_sag_csv(filepath, r_table, sag_table, header=""):
    """Write sag profile to CSV (r_mm, sag_um)."""
    with open(filepath, 'w') as f:
        f.write(f"# {header}\n")
        f.write("# r_mm, sag_um\n")
        for r, s in zip(r_table, sag_table):
            f.write(f"{r * 1e3:.6f}, {s * 1e6:.4f}\n")
    return filepath


def generate_monolithic_stl(design_result, wall_thickness=0.5e-3,
                            radial_segments=100, azimuth_segments=180):
    """
    Generate a monolithic two-lens beam shaper as a single STL.

    Structure (bottom to top):
    - Lens 1 flat face at z=0 (substrate interface)
    - Lens 1 glass body
    - Lens 1 aspheric face (inner surface of the void)
    - Conical void (air gap) with structural walls
    - Lens 2 aspheric face (inner surface of the void)
    - Lens 2 glass body
    - Lens 2 flat face at top

    The void is open at the bottom (around lens 1) to allow
    uncured resin drainage during development.

    Returns list of triangle arrays: [outer_shell, inner_void, ...]
    """
    t1 = design_result['t1']
    t2 = design_result['t2']
    sep = design_result['separation']
    r1 = design_result['r']
    sag1 = design_result['sag1']
    r2 = design_result['R']
    sag2 = design_result['sag2']

    ap1 = float(r1[-1])
    ap2 = float(r2[-1])

    # Geometry z-coordinates:
    # z=0: lens 1 flat face (bottom, substrate)
    # z=t1: lens 1 aspheric vertex
    # z=sep-t2: lens 2 aspheric vertex
    # z=sep: lens 2 flat face (top)

    all_tris = []

    # --- Lens 1: solid plano-aspheric ---
    r1_mesh = np.linspace(0, ap1, radial_segments + 1)
    sag1_mesh = np.interp(r1_mesh, r1, sag1)
    # Aspheric face at z = t1 - sag1(r), normals pointing UP (into void)
    asph1_z = t1 - sag1_mesh
    all_tris.append(_sag_surface_tris(r1_mesh, asph1_z, 0.0, azimuth_segments, outward_up=True))
    # Flat face at z=0, normals pointing DOWN
    all_tris.append(_flat_disk_tris(ap1, 0.0, radial_segments + 1, azimuth_segments, outward_up=False))
    # Rim of lens 1
    all_tris.append(_rim_tris(ap1, 0.0, float(asph1_z[-1]), azimuth_segments))

    # --- Lens 2: solid plano-aspheric ---
    r2_mesh = np.linspace(0, ap2, radial_segments + 1)
    sag2_mesh = np.interp(r2_mesh, r2, sag2)
    # Aspheric face at z = (sep-t2) - sag2(R), normals pointing DOWN (into void)
    asph2_z = (sep - t2) - sag2_mesh
    all_tris.append(_sag_surface_tris(r2_mesh, asph2_z, 0.0, azimuth_segments, outward_up=False))
    # Flat face at z=sep, normals pointing UP
    all_tris.append(_flat_disk_tris(ap2, sep, radial_segments + 1, azimuth_segments, outward_up=True))
    # Rim of lens 2
    all_tris.append(_rim_tris(ap2, float(asph2_z[-1]), sep, azimuth_segments))

    # --- Conical wall connecting lens 1 rim to lens 2 rim ---
    # This is a frustum (truncated cone) from (ap1, z=asph1_z[-1]) to (ap2, z=asph2_z[-1])
    thetas = np.linspace(0, 2 * np.pi, azimuth_segments, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    n_wall_z = 20
    z_wall = np.linspace(float(asph1_z[-1]), float(asph2_z[-1]), n_wall_z)
    r_wall = np.linspace(ap1, ap2, n_wall_z)

    wall_tris = []
    for j in range(n_wall_z - 1):
        for k in range(azimuth_segments):
            k2 = (k + 1) % azimuth_segments
            bl = np.array([r_wall[j] * cos_t[k], r_wall[j] * sin_t[k], z_wall[j]])
            br = np.array([r_wall[j] * cos_t[k2], r_wall[j] * sin_t[k2], z_wall[j]])
            tl = np.array([r_wall[j+1] * cos_t[k], r_wall[j+1] * sin_t[k], z_wall[j+1]])
            tr = np.array([r_wall[j+1] * cos_t[k2], r_wall[j+1] * sin_t[k2], z_wall[j+1]])
            wall_tris.append([bl, tr, br])  # outward normals
            wall_tris.append([bl, tl, tr])
    all_tris.append(np.array(wall_tris, dtype=np.float32))

    return np.concatenate(all_tris, axis=0)
