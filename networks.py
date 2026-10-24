import os
import sys
import torch
import torch.nn as nn
from utils.mesh import (
    create_aligned_basis, vertices_to_faces, update_J, sparse_bmm, PoissonSolverLayer
)

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_project_root, 'poissonnet'))

from poissonnet.networks.PoissonNet import *
import poissonnet.networks.PoissonNet as _pnet_module


def get_simple_mlp(input_dim, output_dim, hidden_dim, n_hidden, device='cuda', skip_linear=False):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim, bias=True),
        torch.nn.LayerNorm(hidden_dim),
        torch.nn.GELU(),
        *[torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU()) for _ in range(n_hidden)],
        *([torch.nn.Linear(hidden_dim, output_dim, bias=False)] if not skip_linear else []),
    ).to(device)


class RotationNet(torch.nn.Module):
    """Predicts a per-face rotation basis via ∇f(S)."""

    def __init__(self, feature_dim, network_type, **kwargs):
        super().__init__()
        self.feat_dim = feature_dim
        self.network_type = network_type
        self.use_edges = False
        self.input_dim = feature_dim
        self.output_dim = 1

        assert network_type in [
            'linear', 'gt'
        ]
        match network_type:
            case 'linear':
                self.mlp = torch.nn.Linear(self.input_dim, self.output_dim, bias=False)
            case 'gt' :
                self.mlp = None

    @staticmethod
    def static_prediction_to_basis(pred_raw, B):
        batch_size, num_faces = B.shape[:2]
        B_flat = B.view(batch_size * num_faces, 3, 3)
        pred_raw_flat = pred_raw.view(batch_size * num_faces, -1)
        basis = create_aligned_basis(pred_raw_flat, B_flat)
        basis = basis.view(batch_size, num_faces, 3, 3)
        return pred_raw, basis

    def prediction_to_basis(self, pred_raw, B):
        return RotationNet.static_prediction_to_basis(pred_raw, B)

    def forward(self, G, features, B, **kwargs):
        batch_size, num_faces = B.shape[:2]

        match self.network_type:
            case 'gt':
                self.basis = B
                return None, B
            case _:
                _feat = self.mlp(features)

        pred_raw = sparse_bmm(G, _feat).reshape(batch_size, num_faces, -1)
        xi, out = self.prediction_to_basis(pred_raw, B)
        self.basis = out
        return xi, out

class ReNJFHead(nn.Module):
    """Rotation-equivariant NJF head."""

    def __init__(self, C_in, C_width, network_type='linear', 
                 complex=True, jacobian_n_hidden=3, naive=False):
        super().__init__()
        self.complex = complex
        self.network_type = network_type
        self.naive = naive

        self.rotation_net = RotationNet(
            feature_dim=C_in,
            network_type=network_type,
        )
        if complex:
            self.grad_mlp = ComplexMLP(in_c=C_width, out_c=3, width=C_width,
                                       num_layers=jacobian_n_hidden + 2, modulate=True)
        else:
            self.jacobian_mlp = get_simple_mlp(C_width, 9, C_width, jacobian_n_hidden)

    def forward(self, pose_features, base_features, G, G3, M, M3, solver, faces,
                vertex_mass, B_original, original_grads=None, edges=None, verts=None):
        B = pose_features.shape[0]
        F = faces.shape[1]

        if not self.naive:
            _, basis = self.rotation_net(G3, base_features, B_original, verts=verts)

        if self.complex:
            grads_2d = torch.bmm(G, pose_features)
            if not self.naive:
                M_rot = torch.bmm(
                    basis.view(B * F, 3, 3)[:, :, :2].transpose(1, 2),
                    B_original.view(B * F, 3, 3)[:, :, :2],
                )
                grads_2d = torch.bmm(
                    M_rot,
                    grads_2d.reshape(B * F, 2, -1),
                ).view(B, F * 2, -1)
            x_faces = vertices_to_faces(pose_features, faces)
            grads_2d = self.grad_mlp(grads_2d, x_faces)
            J_2d = grads_2d.view(B, F, 2, 3)
            J = torch.cat([J_2d, torch.zeros(B, F, 1, 3, device=J_2d.device)], dim=2).view(B, F * 3, 3)
        else:
            face_features = vertices_to_faces(pose_features, faces)
            J = self.jacobian_mlp(face_features).reshape(B, F * 3, 3)

        if not self.naive:
            J = update_J(basis, J, two_d=False)

        if original_grads is not None:
            J = J + original_grads

        rhs = sparse_bmm(G3.transpose(1, 2), M3.unsqueeze(-1) * J)
        u = torch.empty_like(rhs)
        for b in range(B):
            u[b] = PoissonSolverLayer.apply(solver[b], rhs[b].contiguous())
        u = u - torch.sum(u * vertex_mass.unsqueeze(-1), dim=1, keepdim=True) / \
            torch.sum(vertex_mass, dim=1, keepdim=True).unsqueeze(-1)

        return u, J

class CustomPoissonNetv2(PoissonNet):
    def __init__(self, C_in, C_out, C_width=128, n_blocks=4, head=None, extra_features=0,
                 outputs_at='vertices', last_activation=nn.Identity(), config={},
                 abs_grad=False, **kwargs):
        self.use_re_njf = (head == 're_njf')
        _head = 'linear' if self.use_re_njf else head

        self._smal_pose_dim = None
        if isinstance(extra_features, tuple):
            amass_dim, smal_dim = extra_features
            self._smal_pose_dim = smal_dim
            _extra_features = amass_dim
        else:
            _extra_features = extra_features

        super().__init__(
            C_in, C_out, C_width=C_width, n_blocks=n_blocks, head=_head,
            extra_features=0, outputs_at=outputs_at,
            last_activation=last_activation, config=config, **kwargs)

        jacobian_n_hidden = config.get('jacobian_n_hidden', 3)
        naive = config.get('naive', False)
        self.complex_head = config.get('complex', False)
        self.rotation_network_type = config.get('rotation_network_type', 'linear')
        self.n_blocks = n_blocks
        self.abs_grad = abs_grad

        # Handle two cases for extra_features: either an int or a tuple of (amass_dim, smal_dim)
        self.extra_feat_mlp = get_simple_mlp(_extra_features + C_width, C_width, C_width, 0, skip_linear=False)
        if self._smal_pose_dim is not None:
            self.extra_feat_mlp_smal = get_simple_mlp(self._smal_pose_dim + C_width, C_width, C_width, 0, skip_linear=False)

        # Custom rotation equivariant NJF head
        if self.use_re_njf:
            self.re_njf_head = ReNJFHead(C_width, C_width,
                                          network_type=self.rotation_network_type,
                                          complex=self.complex_head,
                                          jacobian_n_hidden=jacobian_n_hidden, naive=naive)

    def forward(self, x_in, M, G, solver, faces, vertex_mass, extra_features=None,
                verts=None, G3=None, M3=None, **kwargs):
        grad_in = None
        if not self.abs_grad:
            with torch.no_grad():
                if self.use_re_njf:
                    grad_in = torch.bmm(G3, verts if verts is not None else x_in)
                else:
                    grad_in = torch.bmm(G, verts if verts is not None else x_in)

        # Process features
        x = self.proj_in(x_in)
        for block in self.blocks:
            x, _ = block(x, M, G, solver, faces, vertex_mass, extra_features=None, **kwargs)

        # Embed poses and features
        if extra_features is not None:
            if extra_features.ndim == 2:
                extra_features = extra_features.unsqueeze(1).expand(-1, x_in.shape[1], -1)
            if self._smal_pose_dim is not None and extra_features.shape[-1] == self._smal_pose_dim:
                extra_features = self.extra_feat_mlp_smal(torch.concat([extra_features, x], dim=-1))
            else:
                extra_features = self.extra_feat_mlp(torch.concat([extra_features, x], dim=-1))
            extra_features = extra_features.contiguous()

        # Original NJF heads, only used for ablations
        if not self.use_re_njf:
            x = extra_features
            if self.head_type == 'njf':
                out, grads = self.head(x, M, G, solver, faces, vertex_mass, original_grads=grad_in)
                return out, grads
            if self.head_type == 'linear':
                out = self.head(x)
            elif self.head_type == 'mlp':
                out = self.head_out(self.head(x))
            out = output_at(out, faces, vertex_mass, domain=self.outputs_at)
            out = self.last_act(out)
            return out

        base_features = x
        pose_features = extra_features if extra_features is not None else None
        B_original = kwargs.get('B')
        result = self.re_njf_head(pose_features, base_features, G, G3, M, M3, solver, faces,
                                   vertex_mass, B_original, original_grads=grad_in,
                                   verts=verts)
        self.basis = getattr(self.re_njf_head.rotation_net, 'basis', None)
        return result


# Copy from PoissonNet, just used to modify the number of layers to be comparable to ReNJFHead
class NJFHeadPatched(nn.Module):
    def __init__(self, in_c, out_c, width, config={}):
        super().__init__()
        n_hidden = config.get('jacobian_n_hidden', 3)
        cmlp_nlayers = n_hidden + 2
        cmlp_modulate = config.get('cmlp_modulate', False)
        self.grad_mlp = ComplexMLP(in_c=in_c, out_c=out_c, width=width, num_layers=cmlp_nlayers, modulate=cmlp_modulate)

    def forward(self, x_in, M, G, solver, faces, vertex_mass, original_grads=None, **kwargs):
        B, V, C = x_in.shape
        F = M.shape[1] // 2

        grads_in = torch.bmm(G, x_in) # (B, 2F, C)
        x_faces = vertices_to_faces(x_in, faces) # (B, F, C)
        grads = self.grad_mlp(grads_in, x_faces) # (B, 2F, C)
        if original_grads is not None:
            grads = grads + original_grads

        # Solve Poisson equation Lu = ∇^T @ face_areas * grads
        rhs = torch.bmm(G.transpose(1, 2), M.unsqueeze(-1) * grads) # (B, V, C)
        u = torch.empty_like(rhs)
        for b in range(B):
            u[b] = PoissonSolver.apply(solver[b], rhs[b].contiguous())
        
        # nullify area-weighted mean:
        u = u - torch.sum(u * vertex_mass.unsqueeze(-1), dim=1, keepdim=True) / torch.sum(vertex_mass, dim=1, keepdim=True).unsqueeze(-1)
        return u, grads

NJFHead = NJFHeadPatched
_pnet_module.NJFHead = NJFHeadPatched