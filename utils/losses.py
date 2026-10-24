import torch
from operators import local_basis


def _weighted_loss(pred, tgt, mass, loss_func, weighted=True):
    sample_loss = loss_func(pred, tgt)
    if weighted:
        return (
            (mass.unsqueeze(-1) * sample_loss).sum(dim=(1, 2)) /
            (mass.sum(dim=1) * pred.shape[-1])
        ).mean()
    return sample_loss.mean()


def l2_loss(pred, tgt, mass, weighted=True):
    def _l2(x, y): return (x - y).pow(2)
    return _weighted_loss(pred, tgt, mass, _l2, weighted=weighted)


def l1_loss(pred, tgt, mass, weighted=True):
    def _l1(x, y): return (x - y).abs()
    return _weighted_loss(pred, tgt, mass, _l1, weighted=weighted)


@torch.no_grad()
def compute_metrics(pred_v, pred_grad, tar_v, G, faces):
    """
    Args:
        pred_v:     (B, V, 3) predicted vertices
        pred_grad:  (B, 2F/3F, 3) predicted Jacobians
        tar_v:      (B, V, 3) target vertices
        G:          (B, 2F/3F, V) gradient operator (sparse or dense)
        faces:      (B, F, 3) face indices

    Returns:
        l2_v, l2_v_std, l2_j, l2_j_std, l2_n, l2_n_std
    """
    B = pred_v.shape[0]
    nF = faces.shape[1]

    tar_centroid = tar_v.mean(dim=1, keepdim=True)
    pred_c = pred_v - pred_v.mean(dim=1, keepdim=True)
    tar_c = tar_v - tar_centroid
    tar_radius = tar_c.norm(dim=-1).max(dim=1).values.view(B, 1, 1)
    pred_c = pred_c / tar_radius
    tar_c = tar_c / tar_radius

    per_vertex_l2 = (pred_c - tar_c).norm(dim=-1)
    l2_v = per_vertex_l2.mean() * 1e2
    l2_v_std = per_vertex_l2.std() * 1e2

    tar_grad = torch.bmm(G, tar_v)
    diff_J = (pred_grad - tar_grad).reshape(B, nF, -1)
    per_face_frob = diff_J.norm(dim=-1)
    l2_j = per_face_frob.mean() * 10
    l2_j_std = per_face_frob.std() * 10

    pred_n = local_basis(pred_v, faces)[..., -1]
    tar_n = local_basis(tar_v, faces)[..., -1]

    cos_angle = (tar_n * pred_n).sum(dim=-1).clamp(-1.0, 1.0)
    per_face_angle = torch.rad2deg(torch.acos(cos_angle))
    l2_n = per_face_angle.mean()
    l2_n_std = per_face_angle.std()

    return l2_v, l2_v_std, l2_j, l2_j_std, l2_n, l2_n_std
