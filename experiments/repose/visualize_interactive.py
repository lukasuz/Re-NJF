"""
Usage:
    python -m experiments.repose.visualize_interactive \
        --results_dir results/trained_model/ours/ \
        --mesh data/demo/spot.obj
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import trimesh
import polyscope as ps
import polyscope.imgui as psim

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from utils import (_euler_deg_to_aa, center_target,
                   rotation_matrix_xyz, rotation_matrix_y, get_test_sequences, load_config,
                   load_network, resample_mesh)
from operators import construct_mesh_operators


# SMPL 21 body joints
SMPL_BONE_GROUPS = [
    ("Spine & Head", [
        ("Spine 1", 2), ("Spine 2", 5), ("Spine 3", 8),
        ("Neck",   11), ("Head",   14),
    ]),
    ("Left Arm", [
        ("L. Collar",   12), ("L. Shoulder", 15),
        ("L. Elbow",    17), ("L. Wrist",    19),
    ]),
    ("Right Arm", [
        ("R. Collar",   13), ("R. Shoulder", 16),
        ("R. Elbow",    18), ("R. Wrist",    20),
    ]),
    ("Left Leg", [
        ("L. Hip",   0), ("L. Knee",  3),
        ("L. Ankle", 6), ("L. Foot",  9),
    ]),
    ("Right Leg", [
        ("R. Hip",   1), ("R. Knee",  4),
        ("R. Ankle", 7), ("R. Foot", 10),
    ]),
    ("Hands", [
        ("L. Hand", 21), ("R. Hand", 22),
    ]),
]

# SMAL 35 joints
SMAL_BONE_GROUPS = [
    ("Head & Neck", [
        ("Neck", 15), ("Head", 16),
        ("Jaw",  32), ("L. Ear", 33), ("R. Ear", 34),
    ]),
    ("Spine", [
        ("Spine 1", 1), ("Spine 2", 2), ("Spine 3", 3),
        ("Spine 4", 4), ("Spine 5", 5), ("Thorax",  6),
    ]),
    ("Front Left Leg", [
        ("L. Shoulder", 7), ("L. Upper", 8),
        ("L. Lower",    9), ("L. Paw",  10),
    ]),
    ("Front Right Leg", [
        ("R. Shoulder", 11), ("R. Upper", 12),
        ("R. Lower",    13), ("R. Paw",  14),
    ]),
    ("Back Left Leg", [
        ("L. Hip",   17), ("L. Upper", 18),
        ("L. Lower", 19), ("L. Paw",  20),
    ]),
    ("Back Right Leg", [
        ("R. Hip",   21), ("R. Upper", 22),
        ("R. Lower", 23), ("R. Paw",  24),
    ]),
    ("Tail", [
        ("Tail 1", 25), ("Tail 2", 26), ("Tail 3", 27),
        ("Tail 4", 28), ("Tail 5", 29), ("Tail 6", 30), ("Tail 7", 31),
    ]),
]

MESH_COLOR = (200 / 255., 200 / 255., 200 / 255.)
BASIS_COLORS = [
    (235 / 255., 77 / 255., 75 / 255.),   # red
    (106 / 255., 176 / 255., 76 / 255.),   # green
    (75 / 255., 135 / 255., 235 / 255.),   # blue
]
VECTOR_RADIUS = 0.001
VECTOR_LENGTH = 0.01


class InteractiveViewer:
    _SMAL_DEFAULT_Y = -90.0

    def __init__(
        self,
        models, configs, device,
        src_verts, faces, vertex_features,
        results_dirs=None, ssaa=2,
    ):
        self.models = models
        self.configs = configs
        self.config = configs[0]
        self.device = device
        self.ssaa = ssaa
        self.src_verts = src_verts
        self.faces = faces
        self.vertex_features = vertex_features
        self.results_dirs = results_dirs
        self.N_models = len(models)
        self.allow_mode_switch = True

        src_np = src_verts[0].cpu().numpy()
        bbox_x = src_np[:, 0].max() - src_np[:, 0].min()
        bbox_z = src_np[:, 2].max() - src_np[:, 2].min()
        self.side_offset_base = float(max(bbox_x, bbox_z) / 2)

        print("Loading AMASS test sequences...")
        try:
            self.test_sequences = get_test_sequences(self.config['amass_path'])
        except Exception as e:
            print(f"Error loading AMASS test sequences: {e}")
            self.test_sequences = []
        print(f"Loaded {len(self.test_sequences)} test sequences")
        self.seq_combo_items = ["Manual (off)"] + [f"Seq {i}" for i in range(len(self.test_sequences))]

        # Viewer state
        self.mode = 'smal'
        self.bone_angles_smpl = [[0.0, 0.0, 0.0] for _ in range(23)]
        self.bone_angles_smal = [[0.0, 0.0, 0.0] for _ in range(35)]
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        self.side_multiplier = 2.0
        self.mesh_level = 0
        self.noise_sigma = 0.0
        self.seq_idx = -1
        self.frame_idx = 0
        self.is_playing = False
        self.playback_fps = 10.0
        self.last_frame_t = 0.0
        self.baked_src = None
        self.do_save = False
        self.show_skeleton = True
        self.hide_source = False
        self.cached_level = None
        self.cached_mesh = None
        self.cached_noise = None
        self.prev_state = None
        self._noise_changed = False

    def run(self):
        ps.init()
        ps.set_SSAA_factor(self.ssaa)
        ps.set_ground_plane_mode("shadow_only")
        ps.set_user_callback(self.callback)
        ps.show()

    @torch.no_grad()
    def callback(self):
        changed, do_bake = self._build_ui()

        cur_state = (
            self.mode,
            tuple(tuple(b) for b in self.bone_angles_smpl),
            tuple(tuple(b) for b in self.bone_angles_smal),
            self.angle_x, self.angle_y, self.angle_z,
            self.side_multiplier, self.mesh_level,
            self.noise_sigma,
            self.seq_idx, self.frame_idx,
            self.show_skeleton,
            self.hide_source,
        )
        if not changed and self.prev_state == cur_state:
            return
        self.prev_state = cur_state

        sub_verts, sub_faces, sub_features, cur_faces_b, operators = self._prepare_mesh()
        cond_feat = self._build_conditioning()
        preds = self._run_forward_pass(sub_verts, sub_features, cur_faces_b, operators, cond_feat)
        basis_np = self._get_basis_np()

        if do_bake:
            self._bake(preds[0], sub_faces, sub_features)

        if self.do_save:
            self.do_save = False
            self._save(sub_verts, sub_faces, preds, basis_np)

        self._update_viz(sub_verts, sub_faces, preds, basis_np)

    ######### UI

    def _build_ui(self) -> tuple:
        """Build all UI widgets. Returns (changed, do_bake)."""
        changed = False

        if self.allow_mode_switch:
            label = "Switch to SMAL" if self.mode == 'smpl' else "Switch to SMPL"
            if psim.Button(label):
                self.mode = 'smal' if self.mode == 'smpl' else 'smpl'
                changed = True
            psim.SameLine()
            psim.TextUnformatted(f"Current mode: {self.mode.upper()}")

        c_rx, self.angle_x = psim.SliderFloat("Rotation X (deg)", self.angle_x, -180.0, 180.0)
        c_ry, self.angle_y = psim.SliderFloat("Rotation Y (deg)", self.angle_y, -180.0, 180.0)
        c_rz, self.angle_z = psim.SliderFloat("Rotation Z (deg)", self.angle_z, -180.0, 180.0)
        c_sm, self.side_multiplier = psim.SliderFloat("Side offset multiplier", self.side_multiplier, 0.5, 5.0)
        c_sl, self.mesh_level     = psim.SliderInt("Mesh resolution level", self.mesh_level, -3, 3)
        c_ns, self.noise_sigma    = psim.SliderFloat("Gaussian Noise ", self.noise_sigma, 0.0, 0.01)
        changed = changed or c_rx or c_ry or c_rz or c_sm or c_sl or c_ns
        self._noise_changed = bool(c_ns)

        do_bake = False
        if psim.Button("Bake"):
            do_bake = True
            changed = True
        psim.SameLine()
        if psim.Button("Save Meshes"):
            self.do_save = True
            changed = True
        psim.SameLine()
        if psim.Button("Reset"):
            self._reset()
            changed = True
        psim.SameLine()
        c_hs, self.hide_source = psim.Checkbox("Disable Source", self.hide_source)
        changed = changed or c_hs

        if self.mode == 'smpl':
            _changed = self._build_smpl_ui()
        else:
            _changed = self._build_smal_ui()
        changed = changed or _changed

        return changed, do_bake

    def _build_smpl_ui(self) -> bool:
        changed = False
        c_seq, combo_sel = psim.Combo("Test Sequence", self.seq_idx + 1, self.seq_combo_items)
        if c_seq:
            self.seq_idx = combo_sel - 1
            self.frame_idx = 0
            self.is_playing = False
            changed = True
        if self.seq_idx >= 0:
            seq_len = len(self.test_sequences[self.seq_idx])
            if self.is_playing:
                now = time.time()
                if now - self.last_frame_t >= 1.0 / self.playback_fps:
                    self.frame_idx = (self.frame_idx + 1) % seq_len
                    self.last_frame_t = now
                    changed = True
            c_fr, self.frame_idx = psim.SliderInt("Frame", self.frame_idx, 0, seq_len - 1)
            changed = changed or c_fr
            play_label = "Pause" if self.is_playing else "Play"
            if psim.Button(play_label):
                self.is_playing = not self.is_playing
                self.last_frame_t = time.time()
                changed = True
            psim.SameLine()
            _, self.playback_fps = psim.SliderFloat("FPS", self.playback_fps, 1.0, 60.0)
        else:
            changed = changed or self._bone_angle_tree(self.bone_angles_smpl, SMPL_BONE_GROUPS)
        return changed

    def _build_smal_ui(self) -> bool:
        return self._bone_angle_tree(self.bone_angles_smal, SMAL_BONE_GROUPS)

    def _bone_angle_tree(self, bone_angles, bone_groups) -> bool:
        changed = False
        _mba = 60
        if psim.TreeNode("Bone Angles (deg)"):
            for group_name, bones in bone_groups:
                if psim.TreeNode(group_name):
                    for bone_name, bone_idx in bones:
                        if psim.TreeNode(bone_name):
                            cx, bone_angles[bone_idx][0] = psim.SliderFloat(f"x##{bone_idx}", bone_angles[bone_idx][0], -_mba, _mba)
                            cy, bone_angles[bone_idx][1] = psim.SliderFloat(f"y##{bone_idx}", bone_angles[bone_idx][1], -_mba, _mba)
                            cz, bone_angles[bone_idx][2] = psim.SliderFloat(f"z##{bone_idx}", bone_angles[bone_idx][2], -_mba, _mba)
                            changed = changed or cx or cy or cz
                            psim.TreePop()
                    psim.TreePop()
            psim.TreePop()
        return changed

    ######### Button actions

    def _reset(self):
        """Reset all parameters to default."""
        for i in range(23):
            self.bone_angles_smpl[i] = [0.0, 0.0, 0.0]
        for i in range(35):
            self.bone_angles_smal[i] = [0.0, 0.0, 0.0]
        self.angle_x = 0.0
        self.angle_y = self._SMAL_DEFAULT_Y if self.mode == 'smal' else 0.0
        self.angle_z = 0.0
        self.side_multiplier = 2.0
        self.mesh_level = 0
        self.noise_sigma = 0.0
        self.baked_src = None
        self.cached_level = None
        self.cached_mesh = None

    def _bake(self, v_pred_centered, sub_faces, sub_features):
        """ 'Bake' the current pose into the mesh."""
        for i in range(23):
            self.bone_angles_smpl[i] = [0.0, 0.0, 0.0]
        for i in range(35):
            self.bone_angles_smal[i] = [0.0, 0.0, 0.0]
        self.angle_x = self.angle_z = 0.0
        self.angle_y = self._SMAL_DEFAULT_Y if self.mode == 'smal' else 0.0
        baked_v = v_pred_centered.clone()
        baked_v = baked_v - baked_v.mean(dim=1, keepdim=True)
        baked_radius = baked_v.norm(dim=-1).max(dim=1, keepdim=True).values.unsqueeze(-1)
        baked_v = baked_v / baked_radius
        baked_feat = baked_v if self.config.get('use_xyz', False) else sub_features.clone()
        self.baked_src = (baked_v, sub_faces.clone(), baked_feat)
        self.mesh_level = 0
        self.cached_level = None
        self.cached_mesh = None
        self.prev_state = None  # force redraw next frame with new source

    def _save(self, sub_verts, sub_faces, preds, basis_np):
        """Save current meshes to PLY files."""
        if self.results_dirs is None:
            return
        save_dir_base = os.path.join(self.results_dirs[0], "interactive_meshes")
        os.makedirs(save_dir_base, exist_ok=True)
        idx = 0
        while os.path.exists(os.path.join(save_dir_base, str(idx))):
            idx += 1
        save_dir = os.path.join(save_dir_base, str(idx))
        os.makedirs(save_dir)

        src_np = sub_verts[0].cpu().numpy()
        src_np = src_np - src_np.mean(axis=0, keepdims=True)
        src_mesh = trimesh.Trimesh(vertices=src_np, faces=sub_faces.cpu().numpy(), process=False)
        if basis_np is not None:
            src_mesh.face_attributes['basis_col0'] = basis_np[:, :, 0]
            src_mesh.face_attributes['basis_col1'] = basis_np[:, :, 1]
            src_mesh.face_attributes['basis_col2'] = basis_np[:, :, 2]
        src_mesh.export(os.path.join(save_dir, 'source.ply'))

        for i, p in enumerate(preds):
            pred_np = p[0].cpu().numpy()
            pred_np = pred_np - pred_np.mean(axis=0, keepdims=True)
            run_name = os.path.basename(os.path.normpath(self.results_dirs[i]))
            fname = f'{run_name}.ply'
            trimesh.Trimesh(vertices=pred_np, faces=sub_faces.cpu().numpy(), process=False
                            ).export(os.path.join(save_dir, fname))

        print(f"Saved meshes to {save_dir}")

    ######### Mesh deformation

    def _prepare_mesh(self):
        if self.mesh_level != self.cached_level:
            base_v  = self.baked_src[0] if self.baked_src is not None else self.src_verts
            base_f  = self.baked_src[1] if self.baked_src is not None else self.faces
            base_ft = self.baked_src[2] if self.baked_src is not None else self.vertex_features
            self.cached_mesh = resample_mesh(base_v, base_f, base_ft, self.mesh_level)
            self.cached_level = self.mesh_level
        res_verts, sub_faces, sub_features = self.cached_mesh

        R = rotation_matrix_xyz(self.angle_x, self.angle_y, self.angle_z, self.device)
        sub_verts = res_verts @ R.T

        if self.noise_sigma > 0.0:
            if self._noise_changed or self.cached_noise is None or self.cached_noise.shape != sub_verts.shape:
                self.cached_noise = torch.randn_like(sub_verts) * self.noise_sigma
            sub_verts = sub_verts + self.cached_noise
        else:
            self.cached_noise = None

        cur_faces_b = sub_faces.unsqueeze(0)
        if self.config.get('use_xyz', False):
            sub_features = sub_verts

        operators = construct_mesh_operators(sub_verts, cur_faces_b, get_both=True)
        return sub_verts, sub_faces, sub_features, cur_faces_b, operators

    def _build_conditioning(self):
        """ Build relative rotation angles based on user input or amass test sequence."""
        if self.mode == 'smpl':
            if self.seq_idx >= 0:
                pose_69 = self.test_sequences[self.seq_idx][self.frame_idx].to(self.device)
                poses_72 = torch.zeros(1, 72, device=self.device)
                poses_72[0, 3:] = pose_69
            else:
                aa = np.stack([_euler_deg_to_aa(*self.bone_angles_smpl[i]) for i in range(23)])
                poses_72 = torch.zeros(1, 72, device=self.device)
                poses_72[0, 3:] = torch.tensor(aa.reshape(69), dtype=torch.float32, device=self.device)
            cond_feat = poses_72[:, 3:]
        else: # smal
            pose_105 = torch.zeros(1, 105, device=self.device)
            for j in range(1, 35):
                aa = _euler_deg_to_aa(*self.bone_angles_smal[j])
                pose_105[0, j*3:j*3+3] = torch.tensor(aa, device=self.device)
            cond_feat = pose_105
        return cond_feat

    def _run_forward_pass(self, sub_verts, sub_features, cur_faces_b, operators, cond_feat):
        """ Run a forward pass on each model."""
        vertex_mass, solvers, G, G3, M, M3, B = operators
        preds = []
        for m_i, c_i in zip(self.models, self.configs):
            x_in = sub_verts if c_i.get('use_xyz', False) else sub_features
            v_pred, _ = m_i(
                x_in=x_in,
                M=M, G=G, solver=solvers, faces=cur_faces_b, vertex_mass=vertex_mass,
                extra_features=cond_feat,
                verts=sub_verts,
                G3=G3, M3=M3, B=B,
            )
            preds.append(center_target(v_pred, vertex_mass))
        return preds

    def _get_basis_np(self):
        basis = getattr(self.models[0], 'basis', None)
        if basis is not None:
            return basis[0].cpu().numpy()
        return None

    ######### Visualization

    def _update_viz(self, sub_verts, sub_faces, preds, basis_np):
        pred_nps   = [p[0].cpu().numpy() for p in preds]
        src_np     = sub_verts[0].cpu().numpy().copy()
        faces_np   = sub_faces.cpu().numpy()
        barycenters = src_np[faces_np].mean(axis=1)

        src_center  = src_np.mean(axis=0, keepdims=True)
        src_np      = src_np      - src_center
        barycenters = barycenters - src_center
        pred_nps    = [p - p.mean(axis=0, keepdims=True) for p in pred_nps]

        src_np[:, 1]       -= src_np[:, 1].min()
        barycenters[:, 1]  -= barycenters[:, 1].min()
        for p in pred_nps:
            p[:, 1] -= p[:, 1].min()

        side_offset = self.side_offset_base * self.side_multiplier
        center = self.N_models / 2.0
        src_x  = (0 - center) * side_offset
        src_np[:, 0]      += src_x
        barycenters[:, 0] += src_x

        ps.remove_all_structures()

        if not self.hide_source:
            ps_src = ps.register_surface_mesh("Source", src_np, faces_np)
            ps_src.set_smooth_shade(True)
            ps_src.set_color(MESH_COLOR)
            ps_src.set_material("wax")

        for i, p in enumerate(pred_nps):
            p[:, 0] += ((i + 1) - center) * side_offset
            name = "Predicted" if self.N_models == 1 else f"Predicted {i}"
            ps_pred = ps.register_surface_mesh(name, p, faces_np)
            ps_pred.set_smooth_shade(True)
            ps_pred.set_color(MESH_COLOR)
            ps_pred.set_material("wax")

        if basis_np is not None and not self.hide_source:
            ps_basis = ps.register_point_cloud("Basis", barycenters,
                                               radius=VECTOR_RADIUS, enabled=True)
            for i in range(3):
                ps_basis.add_vector_quantity(
                    f"Basis {i+1}", basis_np[:, :, i],
                    enabled=i == 0, radius=VECTOR_RADIUS, length=VECTOR_LENGTH,
                    color=BASIS_COLORS[i],
                )


def run_viewer(
    models, configs, device,
    src_verts, faces, vertex_features,
    results_dirs=None, ssaa=2,
):
    viewer = InteractiveViewer(
        models, configs, device,
        src_verts, faces, vertex_features,
        results_dirs=results_dirs, ssaa=ssaa,
    )
    viewer.run()


def _load_from_results_dir(results_dir, device):
    """Load config and weights from a results directory."""
    config = load_config(os.path.join(results_dir, 'config.yaml'))
    for name in ('model_final.pt', 'model_latest.pt'):
        w_path = os.path.join(results_dir, name)
        if os.path.exists(w_path):
            break
    else:
        raise FileNotFoundError(f"No weights found in {results_dir}")
    model, _ = load_network(w_path, config, device)
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Interactive visualizer for repose')
    parser.add_argument('--results_dir', type=str, required=True, nargs='+',
                        help='Path(s) to results directory containing config.yaml and model weights. Pass multiple for comparison mode.')
    parser.add_argument('--mesh', type=str, required=True,
                        help='Path to source mesh .obj file')
    parser.add_argument('--features', type=str, default=None,
                        help='Path to vertex features .pt file (defaults to mesh path with .pt extension)')
    parser.add_argument('--ssaa', type=int, default=2,
                        help='SSAA factor for rendering')
    args = parser.parse_args()

    results_dirs = args.results_dir
    models, configs = [], []
    for rd in results_dirs:
        m, cfg = _load_from_results_dir(rd, torch.device('cpu'))
        models.append(m)
        configs.append(cfg)

    config0 = configs[0]
    device = torch.device(config0.get('device', 'cuda'))
    for m in models:
        m.to(device)

    # Load source mesh
    mesh_path = args.mesh
    feat_path = args.features if args.features is not None else os.path.splitext(mesh_path)[0] + '.pt'
    mesh = trimesh.load(mesh_path, process=False)
    src_verts = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0).to(device)
    faces = torch.tensor(mesh.faces, dtype=torch.long).to(device)
    if "DT4D" in mesh_path:
        faces = faces.flip(-1)
    print(f"Loaded mesh from {mesh_path}: {src_verts.shape[1]} verts, {faces.shape[0]} faces")

    # Normalize to unit sphere
    src_centroid = src_verts.mean(dim=1, keepdim=True)
    src_radius = (src_verts - src_centroid).norm(dim=-1).max(dim=1, keepdim=True).values.unsqueeze(-1)
    src_verts = (src_verts - src_centroid) / src_radius

    # Load vertex features
    ft = torch.load(feat_path, map_location='cpu', weights_only=True)
    vertex_features = ft.float().squeeze().unsqueeze(0).to(device)
    print(f"Loaded features from {feat_path}, shape: {vertex_features.shape}")

    run_viewer(
        models, configs, device,
        src_verts, faces, vertex_features,
        results_dirs=results_dirs, ssaa=args.ssaa,
    )


if __name__ == '__main__':
    main()
