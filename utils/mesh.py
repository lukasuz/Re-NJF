import importlib.util
import numpy as np
import torch
import trimesh
from matplotlib.colors import LinearSegmentedColormap

_error_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F7F7F7", "#1581BF"])
def error_colors(pred, tgt, max_error=0.1):
    err = np.linalg.norm(pred - tgt, axis=-1)
    err = (np.log(1. + err))
    t = np.clip(err / max_error, 0.0, 1.0)
    return np.array(_error_cmap(t))[:, :3]


def norm_pair(src_verts, tar_verts):
    """Normalize src and tar verts to unit sphere centered on src centroid."""
    centroid = src_verts.mean(dim=1, keepdim=True)
    radius = (src_verts - centroid).norm(dim=-1).max(dim=1, keepdim=True).values.unsqueeze(-1)
    return (src_verts - centroid) / radius, (tar_verts - centroid) / radius


@torch.compile()
def sparse_bmm(sparse_3d, dense):
    B, R, C = sparse_3d.shape
    s = sparse_3d.coalesce()
    bi, ri, ci = s.indices()
    flat_sparse = torch.sparse_coo_tensor(
        torch.stack([bi * R + ri, bi * C + ci]), s.values(), (B * R, B * C)
    )
    return (flat_sparse @ dense.reshape(B * C, -1)).reshape(B, R, -1)


def _load_from_submodule(path):
    import sys
    import types
    from pathlib import Path

    path = Path(path)
    module_name = path.stem

    parts = list(path.parts[:-1])
    if parts:
        package_name = '.'.join(parts)
        full_module_name = f"{package_name}.{module_name}"
    else:
        package_name = None
        full_module_name = module_name

    root_dir = str(Path.cwd())
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    for i in range(len(parts)):
        parent_name = '.'.join(parts[:i+1])
        if parent_name not in sys.modules:
            parent_mod = types.ModuleType(parent_name)
            parent_mod.__path__ = [str(Path.cwd() / '/'.join(parts[:i+1]))]
            parent_mod.__package__ = parent_name
            sys.modules[parent_name] = parent_mod

    spec = importlib.util.spec_from_file_location(
        full_module_name, path,
        submodule_search_locations=[str(path.parent)]
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package_name
    sys.modules[full_module_name] = mod
    spec.loader.exec_module(mod)
    return mod


PoissonSolverLayer = _load_from_submodule("poissonnet/networks/common.py").PoissonSolver
def solve_poisson(V, G, M, solvers, J=None, vertex_mass=None, ignore_solver=False, L=None):
    if J is None:
        J = torch.bmm(G, V)
    rhs = sparse_bmm(G.transpose(1, 2), M.unsqueeze(-1) * J)
    u = torch.empty_like(rhs)

    if ignore_solver and L is None:
        G = G.to_dense()
        eps_diag = torch.diag(torch.tensor([1e-6] * V.shape[1])).to(V.device)
        L = torch.bmm(G.transpose(1, 2) * M.unsqueeze(1), G) + eps_diag[None]

    for b in range(V.shape[0]):
        if ignore_solver:
            u[b] = torch.linalg.inv(L[b].double()).float() @ rhs[b]
        else:
            u[b] = PoissonSolverLayer.apply(solvers[b], rhs[b].contiguous())

    if vertex_mass is not None:
        u = u - torch.sum(u * vertex_mass.unsqueeze(-1), dim=1, keepdim=True) / torch.sum(vertex_mass, dim=1, keepdim=True).unsqueeze(-1)
    else:
        u = u - u.mean(dim=1, keepdim=True)
    return u


def update_J(F2, J, two_d=False):
    nB, nF = F2.shape[:2]
    J = J.view(nB * nF, 3, 3)
    F2 = F2.view(nB * nF, 3, 3)
    out = torch.bmm(F2, torch.bmm(J, F2.transpose(1, 2)))
    dim = 3 if not two_d else 2
    if two_d:
        out = torch.bmm(F2[:, :, :2].transpose(2, 1), out)
    out = out.view(nB, nF * dim, 3)
    return out


def faces_to_vertices(face_data, faces, face_areas, num_vertices):
    """Map face data (B, F, C) to vertex data (B, V, C) via area-weighted averaging."""
    B, F, C = face_data.shape
    face_data_expanded = face_data.unsqueeze(2).expand(B, F, 3, C)
    face_data_flat = face_data_expanded.reshape(B, F * 3, C)
    faces_flat = faces.reshape(B, F * 3)
    weights = (face_areas.unsqueeze(-1) / 3).expand(B, F, 3)
    weights_flat = weights.reshape(B, F * 3)
    vertex_data = torch.zeros(B, num_vertices, C, device=face_data.device, dtype=face_data.dtype)
    vertex_mass = torch.zeros(B, num_vertices, device=face_data.device, dtype=face_data.dtype)
    vertex_data = vertex_data.scatter_add(
        dim=1,
        index=faces_flat.unsqueeze(-1).expand(B, F * 3, C),
        src=face_data_flat * weights_flat.unsqueeze(-1)
    )
    vertex_mass = vertex_mass.scatter_add(dim=1, index=faces_flat, src=weights_flat)
    vertex_data = vertex_data / vertex_mass.unsqueeze(-1)
    return vertex_data


def create_aligned_basis(w_projected, B):
    """Create orthonormal basis where e1 aligns with the projected vector."""
    e1 = w_projected / (torch.norm(w_projected, dim=1, keepdim=True) + 1e-8)
    e3 = B[:, :, 2]
    e2 = torch.cross(e3, e1, dim=1)
    e2 = e2 / (torch.norm(e2, dim=1, keepdim=True) + 1e-8)
    return torch.stack([e1, e2, e3], dim=2)


def center_target(tgt, vertex_mass):
    """Center target mesh using mass-weighted centroid."""
    tgt_centroid = (
        torch.sum(tgt * vertex_mass.unsqueeze(-1), dim=1, keepdim=True) /
        torch.sum(vertex_mass, dim=1, keepdim=True).unsqueeze(-1)
    )
    return tgt - tgt_centroid


def rotation_matrix_y(angle_deg, device):
    """Rotation matrix around the y-axis."""
    return rotation_matrix_xyz(0, angle_deg, 0, device)

def rotation_matrix_xyz(angle_x_deg, angle_y_deg, angle_z_deg, device):
    """Combined rotation matrix from XYZ Euler angles (degrees)."""
    ax = torch.deg2rad(torch.tensor(angle_x_deg, dtype=torch.float32))
    ay = torch.deg2rad(torch.tensor(angle_y_deg, dtype=torch.float32))
    az = torch.deg2rad(torch.tensor(angle_z_deg, dtype=torch.float32))
    cx, sx = torch.cos(ax), torch.sin(ax)
    cy, sy = torch.cos(ay), torch.sin(ay)
    cz, sz = torch.cos(az), torch.sin(az)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32, device=device)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32, device=device)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32, device=device)
    return Rz @ Ry @ Rx


def random_rotate_meshes(src_verts, tar_verts):
    """Apply a random SO(3) rotation to src and tar meshes (same rotation per sample)."""
    import roma
    B = src_verts.shape[0]
    R = roma.random_rotmat(B, device=src_verts.device, dtype=src_verts.dtype)
    return src_verts @ R.mT, tar_verts @ R.mT


def _euler_deg_to_aa(x_deg, y_deg, z_deg):
    """XYZ extrinsic Euler angles (degrees) → axis-angle vector (3,)."""
    x, y, z = np.radians(x_deg), np.radians(y_deg), np.radians(z_deg)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    theta = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return (theta * axis).astype(np.float32)

def vertices_to_faces(x, faces):
    # https://github.com/nmwsharp/diffusion-net
    x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
    faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
    xf = torch.gather(x_gather, 1, faces_gather)
    x_out = torch.mean(xf, dim=-1)
    return x_out


def resample_mesh(verts, faces, features, level):
    """
    Modify mesh resolution by `level` steps.
      level > 0: midpoint subdivide `level` times (~4x faces per step)
      level < 0: QEM decimate |level| times (~0.5x faces per step)
      level == 0: no-op

    Args:
        verts:    (1, V, 3)  float32 torch tensor on device
        faces:    (F, 3)     long   torch tensor on device
        features: (1, V, C)  float32 torch tensor on device
        level:    int in [-3, 3]

    Returns:
        new_verts:    (1, V', 3)  on same device
        new_faces:    (F', 3)     on same device
        new_features: (1, V', C)  on same device
    """
    if level == 0:
        return verts, faces, features
    device = verts.device
    v_np = verts[0].cpu().numpy()
    f_np = faces.cpu().numpy()
    feat = features[0].cpu().numpy()  # (V, C)

    if level > 0:
        for _ in range(level):
            v_np, f_np, attrs = trimesh.remesh.subdivide(
                v_np, f_np,
                vertex_attributes={'features': feat},
            )
            feat = attrs['features']
    else:
        for _ in range(-level):
            n_target = max(len(f_np) // 2, 1)
            simplified = trimesh.Trimesh(v_np, f_np, process=False).simplify_quadric_decimation(face_count=n_target, aggression=10)
            diffs = simplified.vertices[:, None] - v_np[None]   # (V', V, 3)
            nn_idx = np.argmin((diffs * diffs).sum(axis=-1), axis=1)
            feat = feat[nn_idx]
            v_np = simplified.vertices
            f_np = simplified.faces

    return (
        torch.tensor(v_np, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(f_np, dtype=torch.long, device=device),
        torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0),
    )