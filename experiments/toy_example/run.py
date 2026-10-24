"""
Complete standalone toy example, that only utilises two simple meshes
and does not require the Cholesky decomposition for the solver. Makes
it slower and only works on smaller meshes but show-cases the main idea.
"""

import argparse
import os
import sys
import numpy as np
import igl
import torch
import trimesh
import imageio
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from utils import render_image, center_target, l2_loss
from matplotlib_constants import WIDTH, LINE_WIDTH, lb_title, lb_tick, lb_legend

device = 'cuda'

########### Helpers

def create_cube(subdivisions=1, radius=0.3):
    """Create a watertight cube mesh of side length 2*radius centered at origin."""
    S = subdivisions
    faces_def = [
        (np.array([0., 1., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.])),
        (np.array([0., 0., 1.]), np.array([0., 1., 0.]), np.array([-1., 0., 0.])),
        (np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])),
        (np.array([1., 0., 0.]), np.array([0., 0., 1.]), np.array([0., -1., 0.])),
        (np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])),
        (np.array([0., 1., 0.]), np.array([1., 0., 0.]), np.array([0., 0., -1.])),
    ]

    coords = np.linspace(-radius, radius, S + 1)
    verts = []
    faces = []
    vert_index = {}

    def add_vertex(p):
        key = tuple(np.round(p, 8))
        if key in vert_index:
            return vert_index[key]
        idx = len(verts)
        vert_index[key] = idx
        verts.append(p)
        return idx

    for u, v, n in faces_def:
        center = n * radius
        grid = np.empty((S + 1, S + 1), dtype=np.int64)
        for i, s in enumerate(coords):
            for j, t in enumerate(coords):
                grid[i, j] = add_vertex(center + u * s + v * t)

        for i in range(S):
            for j in range(S):
                v0 = grid[i, j]
                v1 = grid[i + 1, j]
                v2 = grid[i + 1, j + 1]
                v3 = grid[i, j + 1]
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

    verts = torch.tensor(np.array(verts), device=device, dtype=torch.float32)
    faces = torch.tensor(faces, device=device, dtype=torch.int64)
    return verts, faces


@torch.no_grad()
def to_tensor(arr):
    return torch.tensor(arr, device=device)


@torch.no_grad()
def to_array(tensor):
    return tensor.cpu().numpy()


@torch.no_grad()
def get_grad(vertices_np, faces_np):
    # Transforming between planar and interleaved convention with sparse matrix
    g = igl.grad(vertices_np, faces_np)
    N = len(faces_np)
    perm = np.arange(3 * N).reshape(3, N).T.ravel()
    g = g[perm]
    g = g.tocoo()
    return torch.sparse_coo_tensor(
        torch.tensor(np.array([g.row, g.col]), dtype=torch.long),
        torch.tensor(g.data, dtype=torch.float32),
        size=g.shape,
        dtype=torch.float32,
        device=device
    ).to_dense()


@torch.no_grad()
def get_operators(verts, faces):
    verts_np, faces_np = to_array(verts), to_array(faces)
    G = get_grad(verts_np, faces_np)
    areas = igl.doublearea(verts_np, faces_np) / 2.0
    a = to_tensor(np.repeat(areas, 3)).float()
    L = G.T * a @ G
    eps = 1e-6 * torch.eye(L.shape[0], device=device).float()
    L_inv = torch.linalg.inv((L + eps).double()).float()
    lhs = L_inv @ G.T * a
    M = igl.massmatrix(verts_np, faces_np, igl.MASSMATRIX_TYPE_VORONOI)
    vertex_mass = to_tensor(M.diagonal()).float().unsqueeze(0)  # (1, V)
    face_areas  = to_tensor(np.repeat(areas, 3)).float().unsqueeze(0)  # (1, 3F)
    return G, lhs, vertex_mass, face_areas


@torch.no_grad()
def get_normals(verts, faces):
    normals = to_tensor(igl.per_face_normals(to_array(verts), to_array(faces)))
    return normals.float()


@torch.no_grad()
def load_data(src_path, tgt_path):
    if src_path is None or tgt_path is None:
        src_verts, faces = create_cube(subdivisions=7, radius=0.2)
        print(f"Number of faces: {faces.shape[0]}")
        deformation2 = torch.tensor(
            [[1, 0.8, 0],
            [0, 1, 0],
            [0.8, 0, 1]]).float().cuda()
        tgt_verts = (deformation2 @ src_verts.T).T
    else:
        src_mesh = trimesh.load(src_path, process=False)
        tgt_mesh = trimesh.load(tgt_path, process=False)
        src_verts = to_tensor(np.array(src_mesh.vertices)).float()
        tgt_verts = to_tensor(np.array(tgt_mesh.vertices)).float()
        faces = to_tensor(np.array(src_mesh.faces)).long()

    centroid = src_verts.mean(dim=0)
    radius = (src_verts - centroid).norm(dim=-1).max()
    src_verts = (src_verts - centroid) / radius
    tgt_verts = (tgt_verts - centroid) / radius

    G, lhs, vertex_mass, face_areas = get_operators(src_verts, faces)
    normals = get_normals(src_verts, faces)
    return src_verts, tgt_verts, faces, normals, G, lhs, vertex_mass, face_areas


def create_basis(X_i, N):
    e1 = X_i / (torch.norm(X_i, dim=1, keepdim=True) + 1e-8)
    e2 = torch.cross(N, e1, dim=1)
    e2 = e2 / (torch.norm(e2, dim=1, keepdim=True) + 1e-8)
    return torch.stack([e1, e2, N], dim=2)


def solve_for_vertices(lhs, canonical_J, J, B=None):
    if B is not None:
        _J = J.view(B.shape[0], 3, 3)
        J = (B @ _J @ B.transpose(1, 2)).view(J.shape[0], 3)
    J = canonical_J + J
    verts = lhs @ J
    verts = verts - verts.mean(dim=0, keepdim=True)
    return verts, J


def rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                        dtype=torch.float32, device=device)

########### Training

def build_mlp(feat_dim, hidden_dim, n_hidden):
    return nn.Sequential(
        nn.Linear(feat_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        *[nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU()) for _ in range(n_hidden)
        ],
        nn.Linear(hidden_dim, 9, bias=False),
    ).to(device)


def train_model(naive, src_verts, tgt_verts, faces, normals, G, lhs,
                vertex_mass, face_areas,
                vertex_features, face_features,
                feat_dim=128, hidden_dim=128, n_hidden=4, iters=5000, lr=1e-4):
    F = faces.shape[0]
    d = build_mlp(feat_dim, hidden_dim, n_hidden)
    sigma = nn.Linear(feat_dim, 1).to(device)
    optimizer = torch.optim.AdamW(list(d.parameters()) + list(sigma.parameters()), lr=lr)

    with torch.no_grad():
        M_3d = face_areas
        tgt_centered = center_target(tgt_verts.unsqueeze(0), vertex_mass)  # (1, V, 3)
        tgt_J = G @ tgt_centered.squeeze(0)
        canonical_J = G @ src_verts

    label = "naive" if naive else "ours"
    pbar = tqdm(range(iters), dynamic_ncols=True, desc=f"Training {label}")
    for _ in pbar:
        optimizer.zero_grad()
        pred_J = d(face_features).view(3 * F, 3)
        if naive:
            B = None
        else:
            X_i = (G @ sigma(vertex_features)).view(F, 3)
            B = create_basis(X_i, normals)
        pred_verts, pred_J = solve_for_vertices(lhs, canonical_J, pred_J, B=B)

        pred_centered = center_target(pred_verts.unsqueeze(0), vertex_mass)
        loss_v = l2_loss(pred_centered, tgt_centered, vertex_mass)
        loss_g = l2_loss(pred_J, tgt_J, M_3d)
        loss = loss_v + loss_g
        loss.backward()
        optimizer.step()
        pbar.set_description(f"[{label}] loss: {loss.item():.6f}")

    return d, sigma


########### Main
parser = argparse.ArgumentParser()
parser.add_argument('--plot_only', action='store_true',
                    help='Skip training/eval and regenerate the plot from saved errors.npz')
parser.add_argument('--skip_render', action='store_true')
args = parser.parse_args()

col_ours = '#6FAF4F'
col_naive = '#D97A2B'

viz_step = 3
feat_dim   = 128
hidden_dim = 128
n_hidden   = 3
iters      = 1000
lr         = 1e-4
skip_render = args.skip_render
src_path = tgt_path = None

angles = np.arange(0, 361, viz_step)
errors_save_path = 'results/toy_example/errors.npz'
os.makedirs('results/toy_example', exist_ok=True)

if args.plot_only:
    data = np.load(errors_save_path)
else:
    frames_dir  = 'results/toy_example/frames'
    os.makedirs(frames_dir, exist_ok=True)

    # Load data
    src_verts, tgt_verts, faces, normals, G, lhs, vertex_mass, face_areas = load_data(src_path, tgt_path)
    V, F = src_verts.shape[0], faces.shape[0]
    faces_np = to_array(faces)

    vertex_features = torch.randn(V, feat_dim, device=device)
    face_features   = vertex_features[faces].mean(dim=1)
    vertex_col= vertex_features
    _min, _max = vertex_col.min(dim=0,keepdim=True).values, vertex_col.max(dim=0,keepdim=True).values
    vertex_col = (vertex_col - _min) / (_max - _min)
    vertex_col = vertex_col[:, :3].cpu().numpy()

    # Train all models

    d_ours,  sigma_ours  = train_model(False, src_verts, tgt_verts, faces, normals,
                                        G, lhs, vertex_mass, face_areas,
                                        vertex_features, face_features,
                                        feat_dim, hidden_dim, n_hidden, iters, lr)
    d_naive, _ = train_model(True,  src_verts, tgt_verts, faces, normals,
                                        G, lhs, vertex_mass, face_areas,
                                        vertex_features, face_features,
                                        feat_dim, hidden_dim, n_hidden, iters, lr)

    # Evaluation across rotations
    errors_naive = []
    errors_ours = []
    frame_paths = []

    print("Evaluating across rotations...")
    with torch.no_grad():
        for angle in tqdm(angles, dynamic_ncols=True, desc="Rendering"):
            R = rot_y(float(angle))
            src_rot = src_verts @ R.T
            tgt_rot = tgt_verts @ R.T

            G_rot, lhs_rot, _, _ = get_operators(src_rot, faces)
            normals_rot    = get_normals(src_rot, faces)
            canonical_J_rot = G_rot @ src_rot

            # Naive model
            pred_J_naive = d_naive(face_features).view(3 * F, 3)
            pred_naive, _ = solve_for_vertices(lhs_rot, canonical_J_rot, pred_J_naive, B=None)

            # Ours model
            pred_J_ours = d_ours(face_features).view(3 * F, 3)
            X_i = (G_rot @ sigma_ours(vertex_features)).view(F, 3)
            B   = create_basis(X_i, normals_rot)
            pred_ours, _ = solve_for_vertices(lhs_rot, canonical_J_rot, pred_J_ours, B=B)

            tgt_rot = tgt_rot - tgt_rot.mean(dim=0)
            pred_naive = pred_naive - pred_naive.mean(dim=0)
            pred_ours = pred_ours - pred_ours.mean(dim=0)

            # Errors
            err_naive = (pred_naive - tgt_rot).norm(dim=-1)
            err_ours = (pred_ours - tgt_rot).norm(dim=-1)
            errors_naive.append((err_naive.mean().item(), err_naive.std().item()))
            errors_ours.append( (err_ours.mean().item(),  err_ours.std().item()))

            # Render frame
            frame_path = os.path.join(frames_dir, f'frame_{angle:03d}.png')
            src_v_np = to_array(src_rot) - to_array(src_rot).mean(axis=0)
            pred_naive_v_np = to_array(pred_naive) - to_array(pred_naive).mean(axis=0)
            pred_ours_v_np = to_array(pred_ours) - to_array(pred_ours).mean(axis=0)
            tgt_v_np = to_array(tgt_rot) - to_array(tgt_rot).mean(axis=0)

            _R = to_array(rot_y(-15))

            if not skip_render:
                render_image([
                    ("src",               src_v_np @ _R.T,                   np.flip(faces_np,-1), vertex_col),
                    ("naive",             pred_naive_v_np @ _R.T,            np.flip(faces_np,-1), col_naive),
                    ("ours",              pred_ours_v_np @ _R.T,             np.flip(faces_np,-1), col_ours),
                    ("tgt",               tgt_v_np @ _R.T,                   np.flip(faces_np,-1), vertex_col),
                ], frame_path, mode='polyscope',
                camera_target=(0., 0, 0.),
                camera_orbit=(0., 20, 5.5))
                frame_paths.append(frame_path)

    # Save errors
    np.savez(errors_save_path,
             errors_naive=np.array(errors_naive),
             errors_ours=np.array(errors_ours))
    print(f"Errors saved: {errors_save_path}")

    if not skip_render:
        # Create video
        video_path = 'results/toy_example/rotation_video.mp4'
        video_params = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '10']

        print(f"Writing video to {video_path}...")
        frames_list = [imageio.imread(p) for p in frame_paths]
        imageio.mimwrite(video_path, frames_list, fps=30, output_params=video_params)
        print(f"Video saved: {video_path}")

# Plot errors
errors_naive            = np.array(errors_naive)
errors_ours             = np.array(errors_ours)

width  = WIDTH / 3
height = width / 1.2

plt.figure(figsize=(width, height))
for means_stds, label, color, va, y_off in [
    (errors_naive, "Naive", col_naive, 'top',    -0.02),
    (errors_ours,  "Ours",  col_ours,  'bottom',  0.0),
]:
    means = means_stds[:, 0]
    rads = np.deg2rad(angles[:len(means)])
    plt.plot(rads, means, color=color, linewidth=LINE_WIDTH)

    mid_idx = np.argmin(np.abs(rads - np.pi))
    plt.text(np.pi, means[mid_idx] + y_off, label,
             color=color, fontproperties=lb_legend,
             ha='center', va=va)

plt.xlabel("Rotation angle (rad)", fontproperties=lb_tick)
plt.title("Mean L2 vertex error",     fontproperties=lb_title)
plt.xlim(0, 2 * np.pi)
plt.margins(x=0)
plt.gca().set_xticks([0, np.pi, 2 * np.pi])
plt.gca().set_xticklabels(["0", "π", "2π"])
plt.xticks(fontproperties=lb_tick)
plt.yticks(fontproperties=lb_tick)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(LINE_WIDTH)
ax.tick_params(width=LINE_WIDTH)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
pdf_path = 'results/toy_example/toy_example_plot.pdf'
plt.savefig(pdf_path)
plt.close()
print(f"Plot saved: {pdf_path}")

