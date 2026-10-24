import math

import imageio
import numpy as np

_BASE_COLOR = (0xf7 / 255, 0xf7 / 255, 0xf7 / 255)

_ps_init_done = False
_ps_init_size = (960, 1080)


def render_image(
    meshes,
    save_path,
    camera_pos=(0., 1.2, -2),
    camera_target=(0.0, 0.32, 0.0),
    camera_orbit=None,
    width=960,
    height=1080,
    **kwargs
):
    """Render a row of mesh panels to a single PNG via Polyscope.

    Args:
        meshes:         list of (name, verts_np, faces_np, colors_np_or_None)
                        colors_np: float32 (V, 3) in [0, 1], or None for base color
        save_path:      output PNG path
        camera_pos:     camera position (x, y, z) in Y-up space
        camera_target:  look-at target (x, y, z); if None, uses the mean
                        center of mass averaged across all input meshes
        camera_orbit:   (azimuth_deg, elevation_deg, distance) — when set,
                        overrides camera_pos.  azimuth=0° puts the camera on
                        the +Z side; azimuth=180° on the -Z side.
                        elevation=0° is horizontal, 90° is directly above.
        width, height:  per-panel resolution in pixels
    """
    # Allow hex color strings (e.g. "#D90479") as shorthand for constant per-vertex colors
    resolved = []
    for name, verts, faces, colors in meshes:
        if isinstance(colors, str):
            h = colors.lstrip("#")
            rgb = np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0
            colors = np.tile(rgb, (verts.shape[0], 1))
        resolved.append((name, verts, faces, colors))
    meshes = resolved

    if camera_target is None:
        camera_target = tuple(
            np.mean([verts.mean(axis=0) for _, verts, _, _ in meshes], axis=0).tolist()
        )

    if camera_orbit is not None:
        az_deg, el_deg, dist = camera_orbit
        if dist <= 0:
            raise ValueError(f"camera_orbit distance must be positive (got {dist}).")
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        tx, ty, tz = camera_target
        camera_pos = (
            tx + dist * math.cos(el) * math.sin(az),
            ty + dist * math.sin(el),
            tz - dist * math.cos(el) * math.cos(az),
        )

    _render_polyscope(meshes, save_path, camera_pos, camera_target, width, height)


def _render_polyscope(meshes, save_path, camera_pos, camera_target, width, height):
    import polyscope as ps
    global _ps_init_done, _ps_init_size

    if not _ps_init_done:
        ps.set_allow_headless_backends(True)
        ps.set_window_size(width, height)
        ps.set_SSAA_factor(4)
        ps.set_ground_plane_mode("shadow_only")
        ps.init()
        _ps_init_done = True
        _ps_init_size = (width, height)

    ps.look_at(camera_pos, camera_target)

    panels = []
    for name, verts, faces, colors in meshes:
        m = ps.register_surface_mesh(name, verts, faces)
        m.set_smooth_shade(True)
        m.set_material("wax")
        if colors is not None:
            m.add_color_quantity("error", colors, enabled=True)
        else:
            m.set_color(_BASE_COLOR)
        ps.frame_tick()
        buf = ps.screenshot_to_buffer(transparent_bg=True)
        panels.append(buf)
        ps.remove_all_structures()

    imageio.imwrite(save_path, np.concatenate(panels, axis=1))
