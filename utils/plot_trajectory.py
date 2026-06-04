import os
from pathlib import Path

import numpy as np
import pyvista as pv

# # Force software rendering before importing pyvista
# os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
# os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
# os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'


pv.OFF_SCREEN = True
from scipy.spatial.transform import Rotation as R


def triad_actor(position, R, scale=0.2):
    """Return 3 arrows representing x, y, z axes (x=red, y=green, z=blue)."""
    arrows = []
    colors = ["red", "green", "blue"]
    for i in range(3):
        vec = R[:, i] * scale
        arrow = pv.Arrow(
            start=position,
            direction=vec,
            tip_length=0.2,
            tip_radius=0.03,
            shaft_radius=0.01,
        )
        arrows.append((arrow, colors[i]))
    return arrows


def animation_fun(
    est: np.ndarray,
    targ: np.ndarray,
    str_name: str,
    fps: int,
    save_path: Path = Path("."),
):
    """
    Animate estimated vs target trajectory with tri-axis arrows.
    Top-down view onto XY plane, fixed bounds.
    """
    assert est.shape == targ.shape
    channels = est.shape[0]
    T = est.shape[1]

    # Pad to 3D if necessary
    if channels < 3:
        pos_pad = np.zeros((3 - channels, T))
        est_pos_3d = np.vstack([est[:channels], pos_pad])
        targ_pos_3d = np.vstack([targ[:channels], pos_pad])
    else:
        est_pos_3d = est[:3]
        targ_pos_3d = targ[:3]

    # Extract positions
    est_pos = est_pos_3d.T
    targ_pos = targ_pos_3d.T

    # --- Setup plotter (off-screen, headless safe) ---
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")

    # Compute bounds with margin
    all_points = np.vstack([est_pos, targ_pos])
    mins = all_points.min(0) - 1
    maxs = all_points.max(0) + 1
    bounds = [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]

    # Fix the bounds so all frames stay within the same box
    plotter.set_focus(
        [(mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2, (mins[2] + maxs[2]) / 2]
    )
    # Top-down view (camera above +Z axis)
    enter = [(mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2, (mins[2] + maxs[2]) / 2]
    range_xy = max(maxs[0] - mins[0], maxs[1] - mins[1])
    range_z = maxs[2] - mins[2]
    # margin = 0.5 * max(range_xy, range_z)
    margin = max(range_xy, range_z)
    camera_height = maxs[2] + range_xy + margin
    plotter.set_position([enter[0], enter[1], camera_height])
    plotter.set_viewup([0, 1, 0])  # Y is up on screen
    plotter.show_bounds(bounds=bounds, grid=True, all_edges=True)

    # Add empty meshes for trajectories
    est_line = pv.PolyData(est_pos[:1])
    plotter.add_mesh(est_line, color="red", line_width=3, label="Estimate")

    targ_line = pv.PolyData(targ_pos[:1])
    plotter.add_mesh(targ_line, color="blue", line_width=3, label="Target")

    # add legend
    plotter.add_legend()
    triad_actors = []

    def update_frame(i):
        nonlocal triad_actors

        n = i + 1
        est_line.points = est_pos[:n]
        est_line.lines = np.hstack([n, np.arange(n)])
        targ_line.points = targ_pos[:n]
        targ_line.lines = np.hstack([n, np.arange(n)])

        # Remove old triad
        for act in triad_actors:
            plotter.remove_actor(act)
        triad_actors = []

        # Orientation triads for both estimation and ground truth
        if channels == 6:
            for data, color in zip(
                [est, targ], ["red", "blue"]
            ):  # Red for estimation, blue for ground truth
                z_axis = data[3:, i]
                z_axis_norm = np.linalg.norm(z_axis)

                # Check if the vector has sufficient magnitude
                if z_axis_norm < 1e-6:
                    # Use default orientation if vector is too small
                    rotation_matrix = np.eye(3)
                else:
                    z_axis_normalized = z_axis / z_axis_norm

                    try:
                        # Create rotation that aligns [0,0,1] with z_axis
                        rotation = R.align_vectors([z_axis_normalized], [[0, 0, 1]])[0]
                        rotation_matrix = rotation.as_matrix()
                    except (ValueError, RuntimeError):
                        # Fallback to identity matrix if alignment fails
                        rotation_matrix = np.eye(3)

                pos_3d = data[:3, i]
                arrows = triad_actor(pos_3d, rotation_matrix, scale=0.2)
                for arrow, arrow_color in arrows:
                    triad_actors.append(plotter.add_mesh(arrow, color=arrow_color))

        # Position error
        pos_error = np.linalg.norm(est_pos[i] - targ_pos[i])
        plotter.add_text(
            f"{str_name}\nFrame {i+1}/{T}\nError: {pos_error:.3f} m",
            position="upper_left",
            font_size=12,
            name="title",
            font="times",
        )

    # Save path
    save_path.mkdir(parents=True, exist_ok=True)
    movie_path = save_path / f"{str_name}.mp4"
    plotter.open_movie(str(movie_path), framerate=fps)

    for i in range(T):
        update_frame(i)
        plotter.write_frame()

    # save the last frame as png file
    # play the last frame and save
    # Play the last frame and save as PNG
    # update_frame(T - 1)
    last_frame_path = save_path / f"{str_name}_last_frame.png"
    plotter.screenshot(str(last_frame_path))

    plotter.close()
    return movie_path


# --- Example usage ---
if __name__ == "__main__":
    # T = 200
    # t = np.linspace(0, 4 * np.pi, T)
    # targ = np.vstack(
    #     [
    #         np.cos(t),
    #         np.sin(t),
    #         0.1 * t,  # target position
    #         np.zeros_like(t),
    #         np.zeros_like(t),
    #         np.ones_like(t),
    #     ]
    # )
    # est = targ + 0.05 * np.random.randn(*targ.shape)  # noisy estimate

    # save_path = Path("./output")
    # path = animation_fun(
    #     est=est,
    #     targ=targ,
    #     str_name="trajectory_topdown",
    #     fps=30,
    #     save_path=save_path,
    # )
    # print(f"✅ Animation saved to: {path}")
    import torch

    root = Path(
        "./logs_diffusionSpectrum3D/diffusion_hybrid/version_0/testing/diffusion_TLIO/version_0/label_TLIO/datasets_TLIO_tlio_golden_145820422949970"
    )
    estimates_path = root / "integral_estimates.pt"
    targets_path = root / "integral_targets.pt"
    est_integrated = torch.load(estimates_path, weights_only=False)
    targ_integrated = torch.load(targets_path, weights_only=False)
    animation_fun(
        est=est_integrated,
        targ=targ_integrated,
        str_name=root.name,
        save_path=root,
        fps=100,
    )
