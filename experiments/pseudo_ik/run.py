"""Interactive pseudo inverse kinematics via.

Usage:
    python -m experiments.pseudo_ik.run \
        --results_dir results/trained_model/ours/ \
        --mesh data/demo/beast.obj \
        --mode smpl
"""

import argparse
import glob
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import trimesh
import polyscope as ps
import polyscope.imgui as psim

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from utils import rotation_matrix_xyz, load_config, solve_poisson, load_network, l1_loss, l2_loss
from operators import construct_mesh_operators


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Interactive IK via pose optimization')
    parser.add_argument('--mesh', type=str, required=True,
                        help='Path to source mesh .obj')
    parser.add_argument('--features', type=str, default=None,
                        help='Path to vertex features .pt (default: source with .pt ext)')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to trained model results dir')
    parser.add_argument('--mode', type=str, default='smpl', choices=['smpl', 'smal'],
                        help='Skeleton mode: smpl (pose_dim=69) or smal (pose_dim=105)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='AdamW learning rate')
    parser.add_argument('--lambda_b', type=float, default=0.,
                        help='Penalizes if vertices deviate from original position.')
    parser.add_argument('--lambda_s', type=float, default=0,
                        help='Enforces sparsity of pose parameters or Jacobians.')
    parser.add_argument('--lambda_a', type=int, default=1,
                        help='ARAP regularizer.')
    parser.add_argument('--max_iters', type=int, default=500,
                        help='')
    parser.add_argument('--steps_per_frame', type=int, default=1,
                        help='Optimization steps per GUI frame tick')
    parser.add_argument('--ball_radius', type=float, default=0.05,
                        help='Initial selection ball radius')
    return parser.parse_args()


@dataclass
class IKState:
    # Ball
    ball_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ball_radius: float = 0.05
    lambda_s: float = 1.0
    lambda_b: float = 1.0
    lambda_a: int = 1
    lr: float = 0.001
    threshold: float = 0.1
    max_iters: float = 1000
    baseline_mode: bool = False
    use_threshold: bool = False

    # Source constraint
    source_indices: Optional[np.ndarray] = None
    source_center: Optional[np.ndarray] = None

    # Target
    target_positions: Optional[torch.Tensor] = None  # (K, 3) on device

    # Optimization
    is_optimizing: bool = False
    opt_step: int = 0
    optimizer: Any = None
    pose_params: Any = None
    last_loss: float = 0.0


class IKViewer:
    MESH_COLOR = (200 / 255., 200 / 255., 200 / 255.)

    def __init__(
        self,
        model, config,
        src_verts_norm, faces_b, faces_np,
        vertex_features,
        device,
        args,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.src_verts_norm = src_verts_norm
        self.faces_b = faces_b
        self.faces_np = faces_np
        self.mesh_path = args.mesh
        self.pose_dim = 105 if args.mode == 'smal' else 69
        self.steps_per_frame = args.steps_per_frame
        self.ik_save_dir = os.path.join("./results", "IK")

        pn_head = config.get('pn_head', 're_njf')
        self.use_3d_ops = (pn_head == 're_njf')
        self.use_xyz = config.get('use_xyz', False)

        # Numpy helpers
        src_np = src_verts_norm[0].cpu().numpy()
        self.n_verts = src_np.shape[0]

        # Mutable visualization state
        self.src_np = src_np.copy()
        self.display_np = src_np.copy()
        self.vertex_features = vertex_features
        self.current_verts = src_verts_norm.clone()
        self.ball_R = np.eye(3, dtype=np.float32)

        # Rotation
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0

        # Mesh operators (recomputed on rotation)
        self.vertex_mass, self.solvers, self.G, self.G3, self.M, self.M3, self.B = \
            construct_mesh_operators(src_verts_norm, faces_b, get_both=True)
        G_op = self.G3 if self.use_3d_ops else self.G
        with torch.no_grad():
            self.src_jac = torch.bmm(G_op, src_verts_norm)

        # Save/load UI state
        self.experiment_name = "experiment"
        self.selected_file_idx = 0

        # IK state
        self.state = IKState(
            ball_center=src_np.mean(axis=0).copy(),
            ball_radius=args.ball_radius,
            lambda_s=args.lambda_s,
            lambda_b=args.lambda_b,
            lambda_a=args.lambda_a,
            lr=args.lr,
            max_iters=args.max_iters,
        )

    ######### Entry point

    def run(self):
        ps.init()
        ps.set_SSAA_factor(2)
        ps.set_ground_plane_mode("shadow_only")

        ps_mesh = ps.register_surface_mesh("Source", self.src_np, self.faces_np)
        ps_mesh.set_smooth_shade(True)
        ps_mesh.set_color(self.MESH_COLOR)
        ps_mesh.set_material("wax")

        self._register_ball(self.state.ball_center, self.state.ball_radius)

        def callback():
            self._build_rotation_ui()
            psim.Separator()
            self._build_selection_ui()
            psim.Separator()
            self._build_optimization_ui()
            psim.Separator()
            self._build_save_load_ui()

        ps.set_user_callback(callback)
        ps.show()

    ######### UI

    def _build_rotation_ui(self):
        """Global rotation."""
        changed_rx, self.angle_x = psim.SliderFloat("Rotation X (deg)", self.angle_x, -180.0, 180.0)
        changed_ry, self.angle_y = psim.SliderFloat("Rotation Y (deg)", self.angle_y, -180.0, 180.0)
        changed_rz, self.angle_z = psim.SliderFloat("Rotation Z (deg)", self.angle_z, -180.0, 180.0)

        if changed_rx or changed_ry or changed_rz:
            self._apply_rotation()

    def _build_selection_ui(self):
        """Ball selection logic and visualization."""
        # GUI elements
        if ps.have_selection():
            sel = ps.get_selection()
            if sel.is_hit and sel.structure_name == "Source":
                self.state.ball_center = np.array(sel.position, dtype=np.float64)
                self._register_ball(self.state.ball_center, self.state.ball_radius)
            ps.reset_selection()

        changed_r, self.state.ball_radius = psim.SliderFloat(
            "Ball Radius", self.state.ball_radius, 0.01, 0.5)
        changed_p, new_pos = psim.SliderFloat3(
            "Ball Position", self.state.ball_center.tolist(), -1, 1)

        if changed_p:
            self.state.ball_center = np.array(new_pos)
        if changed_r or changed_p:
            self._register_ball(self.state.ball_center, self.state.ball_radius)

        selected = self._get_vertices_in_ball(self.state.ball_center, self.state.ball_radius)
        colors = self._build_vertex_colors(
            selected=selected if self.state.source_indices is None else None,
            source=self.state.source_indices,
        )
        ps.get_surface_mesh("Source").add_color_quantity(
            "selection", colors, defined_on='vertices', enabled=True)

        psim.Separator()

        # Source / target selection logic
        psim.TextUnformatted(f"Vertices in ball: {len(selected)}")

        if self.state.source_indices is None:
            if len(selected) > 0 and psim.Button("Mark Source"):
                self.state.source_indices = selected.copy()
                self.state.source_center = self.state.ball_center.copy()
                print(f"Marked {len(selected)} source vertices")
        else:
            psim.TextUnformatted(
                f"Source: {len(self.state.source_indices)} verts. "
                "Move ball to target, then click 'Mark Target'.")

            if psim.Button("Mark Target"):
                current_source_center = self.src_np[self.state.source_indices].mean(axis=0)
                displacement = self.state.ball_center - current_source_center
                src_positions = torch.tensor(
                    self.src_np[self.state.source_indices],
                    dtype=torch.float32, device=self.device)
                self.state.target_positions = src_positions + torch.tensor(
                    displacement, dtype=torch.float32, device=self.device)
                target_center = self.state.target_positions.cpu().numpy().mean(axis=0)
                self._register_ball(target_center, self.state.ball_radius,
                                    color=(0.2, 1.0, 0.3), name="Target Ball")
                print(f"Target set (displacement: {displacement})")

            psim.SameLine()
            if psim.Button("Cancel Source"):
                self.state.source_indices = None
                self.state.source_center = None
                self.state.target_positions = None
                ps.remove_point_cloud("Target Ball", error_if_absent=False)

    def _build_optimization_ui(self):
        """Optimization hyperparameters and logic."""
        # Optimization GUI stuff
        changed_lambda_s, new_lambda_s = psim.SliderFloat("lambda_s", self.state.lambda_s, 0, 2.0)
        changed_lambda_b, new_lambda_b = psim.SliderFloat("lambda_b", self.state.lambda_b, 0, 2.0)
        changed_lambda_a, new_lambda_a = psim.SliderInt("lambda_a", int(self.state.lambda_a), 0, 1000)
        changed_max_iters, new_max_iters = psim.SliderInt("Max iters", self.state.max_iters, 0, 500)
        changed_lr, new_lr = psim.SliderFloat("lr", self.state.lr, 0, 1e-2)
        _, self.state.baseline_mode = psim.Checkbox("Arap-only", self.state.baseline_mode)
        _, self.state.use_threshold = psim.Checkbox("Use Threshold", self.state.use_threshold)

        if changed_lambda_s:
            self.state.lambda_s = new_lambda_s
        if changed_lambda_b:
            self.state.lambda_b = new_lambda_b
        if changed_lambda_a:
            self.state.lambda_a = new_lambda_a
        if changed_lr:
            self.state.lr = new_lr
        if changed_max_iters:
            self.state.max_iters = new_max_iters

        if self.state.target_positions is not None and psim.Button("Optimize"):
            if self.state.baseline_mode:
                self.state.pose_params = torch.nn.Parameter(
                    torch.zeros_like(self.src_jac))
            else:
                self.state.pose_params = torch.nn.Parameter(
                    torch.randn(1, self.pose_dim, device=self.device) * 1e-6)
            self.state.optimizer = torch.optim.AdamW(
                [self.state.pose_params], lr=self.state.lr)
            self.state.is_optimizing = True
            self.state.opt_step = 0
            print(f"Starting optimization (baseline={self.state.baseline_mode})...")

        # Running progress (only when optimizing)
        if self.state.is_optimizing:
            if self.state.opt_step > self.state.max_iters:
                self.state.is_optimizing = False
            else:
                psim.Separator()
                psim.TextUnformatted(
                    f"Step: {self.state.opt_step}  |  Loss: {self.state.last_loss:.6f}")
                _, self.steps_per_frame = psim.SliderInt("Steps/frame", self.steps_per_frame, 1, 20)

                v_pred = self._optimization_step()
                pred_np = v_pred[0].detach().cpu().numpy()
                self.display_np = pred_np
                ps.get_surface_mesh("Source").update_vertex_positions(pred_np)

                if psim.Button("Stop"):
                    self.state.is_optimizing = False
                    print(f"Stopped at step {self.state.opt_step}, loss={self.state.last_loss:.6f}")
                psim.SameLine()
                if psim.Button("Reset & Edit"):
                    self._reset_all()

    def _build_save_load_ui(self):
        """Experiment saving and loading, as well as mesh saving."""
        _, self.experiment_name = psim.InputText("Experiment Name", self.experiment_name)
        if self.state.target_positions is not None:
            if psim.Button("Save Experiment"):
                self._save_experiment(self.experiment_name)
            psim.SameLine()
            if psim.Button("Save Meshes"):
                self._save_meshes(self.config.get('exp_name', 'unknown'))

        psim.Separator()

        saved_files = sorted(glob.glob(os.path.join(self.ik_save_dir, "*.npz")))
        if saved_files:
            names = [os.path.splitext(os.path.basename(f))[0] for f in saved_files]
            _, self.selected_file_idx = psim.Combo("Saved Experiments", self.selected_file_idx, names)
            if psim.Button("Load Experiment"):
                self._load_and_apply_experiment(saved_files[self.selected_file_idx])

        if psim.Button("Reset All"):
            self._reset_all()

    ######### Helpers

    def _apply_rotation(self):
        """Apply global rotation and update operators."""
        R = rotation_matrix_xyz(self.angle_x, self.angle_y, self.angle_z, self.device)
        rotated_verts = self.src_verts_norm @ R.T
        self.current_verts = rotated_verts

        if self.use_xyz:
            self.vertex_features = rotated_verts

        self.vertex_mass, self.solvers, self.G, self.G3, self.M, self.M3, self.B = \
            construct_mesh_operators(rotated_verts, self.faces_b, get_both=True)
        G_op = self.G3 if self.use_3d_ops else self.G
        self.src_jac = torch.bmm(G_op, rotated_verts)

        self.src_np = rotated_verts[0].detach().cpu().numpy()
        self.display_np = self.src_np.copy()
        ps.get_surface_mesh("Source").update_vertex_positions(self.src_np)

    def _reset_all(self):
        """Reset state."""
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        self.ball_R = np.eye(3, dtype=np.float32)
        self.current_verts = self.src_verts_norm.clone()
        if self.use_xyz:
            self.vertex_features = self.src_verts_norm
        self.vertex_mass, self.solvers, self.G, self.G3, self.M, self.M3, self.B = \
            construct_mesh_operators(self.src_verts_norm, self.faces_b, get_both=True)
        G_op = self.G3 if self.use_3d_ops else self.G
        self.src_jac = torch.bmm(G_op, self.src_verts_norm)
        self.src_np = self.src_verts_norm[0].cpu().numpy()
        self.display_np = self.src_np.copy()
        self.state.source_indices = None
        self.state.source_center = None
        self.state.target_positions = None
        self.state.is_optimizing = False
        ps.remove_point_cloud("Target Ball", error_if_absent=False)
        ps.get_surface_mesh("Source").update_vertex_positions(self.src_np)

    def _get_vertices_in_ball(self, center, radius):
        dists = np.linalg.norm(self.display_np - center, axis=1)
        return np.where(dists <= radius)[0]

    def _register_ball(self, center, radius, color=(1.0, 0.8, 0.0), name="Ball"):
        pc = ps.register_point_cloud(name, center.reshape(1, 3), color=color)
        pc.set_radius(radius, relative=False)
        return pc

    def _build_vertex_colors(self, selected=None, source=None):
        colors = np.full((self.n_verts, 3), 0.78)
        if source is not None and len(source) > 0:
            colors[source] = [0.2, 0.4, 1.0]
        if selected is not None and len(selected) > 0:
            colors[selected] = [1.0, 0.2, 0.2]
        return colors

    def _save_experiment(self, name):
        """Save experiment parameters to load for a different config."""
        os.makedirs(self.ik_save_dir, exist_ok=True)
        path = os.path.join(self.ik_save_dir, f"{name}.npz")
        np.savez(
            path,
            source_indices=self.state.source_indices,
            source_center=self.state.source_center,
            target_positions=self.state.target_positions.cpu().numpy(),
            ball_center=self.state.ball_center,
            ball_radius=np.array(self.state.ball_radius),
            rotation_angles=np.array([self.angle_x, self.angle_y, self.angle_z]),
            lambda_s=np.array(self.state.lambda_s),
            lambda_b=np.array(self.state.lambda_b),
            lambda_a=np.array(self.state.lambda_a),
            lr=np.array(self.state.lr),
            max_iters=self.state.max_iters,
            source_mesh_path=np.array(self.mesh_path),
        )
        print(f"Saved experiment to {path}")

    def _load_and_apply_experiment(self, path):
        """Load experiment parameters."""
        data = np.load(path, allow_pickle=True)
        data = {k: data[k] for k in data.files}
        self.state.source_indices = data['source_indices']
        self.state.source_center = data['source_center']
        self.state.target_positions = torch.tensor(
            data['target_positions'], dtype=torch.float32, device=self.device)
        self.state.ball_center = data['ball_center']
        self.state.ball_radius = float(data['ball_radius'])
        self.state.lambda_s = float(data['lambda_s'])
        self.state.lambda_b = float(data['lambda_b'])
        self.state.lambda_a = float(data['lambda_a'])
        self.state.lr = float(data['lr'])
        angles = data['rotation_angles']
        self.angle_x, self.angle_y, self.angle_z = float(angles[0]), float(angles[1]), float(angles[2])

        self._apply_rotation()
        self.ball_R = rotation_matrix_xyz(
            self.angle_x, self.angle_y, self.angle_z, self.device
        ).detach().cpu().numpy().astype(np.float32)

        self._register_ball(self.state.ball_center, self.state.ball_radius)
        target_center = self.state.target_positions.cpu().numpy().mean(axis=0)
        self._register_ball(target_center, self.state.ball_radius,
                            color=(0.2, 1.0, 0.3), name="Target Ball")

    def _save_meshes(self, exp_name):
        """Save meshes for visualization."""
        random_id = uuid.uuid4().hex[:8]
        out_dir = os.path.join(self.ik_save_dir, exp_name, random_id)
        os.makedirs(out_dir, exist_ok=True)

        src_mesh = trimesh.Trimesh(vertices=self.src_np, faces=self.faces_np, process=False)

        source_ball = trimesh.creation.icosphere(subdivisions=3, radius=self.state.ball_radius)
        source_ball.apply_translation(self.state.source_center)
        source_ball.visual.vertex_colors = np.tile(
            [51, 102, 255, 255], (len(source_ball.vertices), 1))

        target_center = self.state.target_positions.cpu().numpy().mean(axis=0)
        target_ball = trimesh.creation.icosphere(subdivisions=3, radius=self.state.ball_radius)
        target_ball.apply_translation(target_center)
        target_ball.visual.vertex_colors = np.tile(
            [51, 255, 51, 255], (len(target_ball.vertices), 1))

        combined = trimesh.util.concatenate([src_mesh, source_ball, target_ball])
        combined.export(os.path.join(out_dir, 'source_with_balls.ply'))

        out_mesh = trimesh.Trimesh(vertices=self.display_np, faces=self.faces_np, process=False)
        out_mesh.export(os.path.join(out_dir, 'output.obj'))
        print(f"Saved meshes to {out_dir}")

    def _optimization_step(self):
        """Optimization logic."""
        state = self.state
        idx_t = torch.tensor(
            state.source_indices, dtype=torch.long, device=self.current_verts.device)
        idx_base = [i for i in range(self.vertex_features.shape[1])
                    if i not in state.source_indices]
        idx_base = torch.tensor(
            idx_base, dtype=torch.long, device=self.current_verts.device)

        G_op = self.G3 if self.use_3d_ops else self.G
        M_op = self.M3 if self.use_3d_ops else self.M

        v_pred = None
        for i in range(self.steps_per_frame):
            base_og_position = self.current_verts[0, idx_base]
            state.optimizer.zero_grad()

            if state.baseline_mode:
                J_final = self.src_jac + state.pose_params
                v_pred = solve_poisson(
                    self.current_verts, G_op, M_op, self.solvers,
                    J=J_final, vertex_mass=self.vertex_mass)
            else:
                if self.pose_dim == 105:
                    with torch.no_grad():
                        state.pose_params[:, :3] = 0

                v_pred, J_pred = self.model(
                    x_in=self.vertex_features,
                    M=self.M, G=self.G, solver=self.solvers, faces=self.faces_b,
                    vertex_mass=self.vertex_mass,
                    extra_features=state.pose_params,
                    verts=self.current_verts,
                    G3=self.G3, M3=self.M3, B=self.B,
                )
                J_final = J_pred

            if i == 0:
                with torch.no_grad():
                    offset = (v_pred - self.current_verts).mean(dim=1, keepdim=True)

            v_pred = v_pred - offset

            with torch.no_grad():
                dist = torch.linalg.norm(
                    v_pred[0, idx_t] - state.target_positions, dim=-1).mean()
                if dist < state.threshold and state.use_threshold:
                    print("Converged, stopping optimization")
                    state.is_optimizing = False

            loss = l2_loss(v_pred[:,idx_t], state.target_positions[None], self.vertex_mass[:,idx_t])
            if state.lambda_b > 0:
                loss = loss + state.lambda_b * l1_loss(v_pred[:,idx_base], base_og_position[None], self.vertex_mass[:,idx_base])
            
            if state.lambda_s > 0:
                if state.baseline_mode:
                    loss = loss + state.lambda_s * l1_loss(state.pose_params, torch.zeros_like(state.pose_params), self.M3 if self.use_3d_ops else self.M)
                else:
                    loss = loss + state.lambda_s * state.pose_params.abs().mean()

            if state.lambda_a > 0:
                try:
                    J_final = J_final.view(-1, 3, 3)
                except RuntimeError:
                    J_final = J_final.view(-1, 2, 3)
                S = torch.linalg.svdvals(J_final)
                loss += state.lambda_a * (S[:, :2] - 1).pow(2).mean()

            loss.backward()
            state.optimizer.step()
            state.opt_step += 1
            state.last_loss = loss.item()

        return v_pred


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    config = load_config(os.path.join(args.results_dir, 'config.yaml'))
    for name in ('model_final.pt', 'model_latest.pt'):
        w_path = os.path.join(args.results_dir, name)
        if os.path.exists(w_path):
            break
    else:
        raise FileNotFoundError(f"No weights found in {args.results_dir}")
    model, _ = load_network(w_path, config, device)

    # Load mesh
    src_mesh = trimesh.load(args.mesh, process=False)
    src_verts = torch.tensor(src_mesh.vertices, dtype=torch.float32, device=device).unsqueeze(0)
    faces = torch.tensor(src_mesh.faces, dtype=torch.long, device=device)
    if "DT4D" in args.mesh:
        faces = faces.flip(-1)
    faces_b = faces.unsqueeze(0)
    faces_np = faces.cpu().numpy()
    print(f"Source: {args.mesh} ({src_verts.shape[1]} verts, {faces.shape[0]} faces)")

    # Normalize to unit sphere
    centroid = src_verts.mean(dim=1, keepdim=True)
    radius = (src_verts - centroid).norm(dim=-1).max(dim=1, keepdim=True).values.unsqueeze(-1)
    src_verts_norm = (src_verts - centroid) / radius

    # Load features
    use_xyz = config.get('use_xyz', False)
    if use_xyz:
        vertex_features = src_verts_norm
    else:
        feat_path = args.features or os.path.splitext(args.mesh)[0] + '.pt'
        ft = torch.load(feat_path, map_location='cpu', weights_only=True)
        vertex_features = ft.float().squeeze().unsqueeze(0).to(device)
        print(f"Features: {feat_path} (shape: {vertex_features.shape})")

    viewer = IKViewer(
        model=model,
        config=config,
        src_verts_norm=src_verts_norm,
        faces_b=faces_b,
        faces_np=faces_np,
        vertex_features=vertex_features,
        device=device,
        args=args,
    )
    viewer.run()


if __name__ == '__main__':
    main()
