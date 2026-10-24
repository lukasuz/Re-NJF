import torch
import cholespy
import torch_mesh_ops as TMO

def apply_sparse_rotation(sparse_matrix, proj_matrices):
    """
    Applies a batched rotation to a sparse matrix without densifying.
    Vectorized via a single large sparse @ sparse matmul.
    """
    device = proj_matrices.device
    B, F = proj_matrices.shape[:2]
    r, c = proj_matrices.shape[2], proj_matrices.shape[3]

    # 1. Build one big block-diagonal P of shape (B*F*r, B*F*c).
    #    B*F blocks of size (r, c) along the diagonal.
    total_blocks = B * F
    block_rows = torch.arange(r, device=device).repeat_interleave(c)
    block_cols = torch.arange(c, device=device).repeat(r)
    row_offsets = torch.arange(total_blocks, device=device) * r
    col_offsets = torch.arange(total_blocks, device=device) * c
    all_rows = (row_offsets.unsqueeze(1) + block_rows.unsqueeze(0)).flatten()
    all_cols = (col_offsets.unsqueeze(1) + block_cols.unsqueeze(0)).flatten()
    all_values = proj_matrices.contiguous().view(-1)

    P = torch.sparse_coo_tensor(
        torch.stack([all_rows, all_cols]),
        all_values,
        (B * F * r, B * F * c)
    )

    # 2. Stack sparse_matrix into (B*F*c, M) by shifting row indices.
    Fc = F * c
    if sparse_matrix.dim() == 2:
        # Shared across batch: replicate B times vertically
        M = sparse_matrix.shape[1]
        sm = sparse_matrix.coalesce()
        sm_idx = sm.indices()
        sm_val = sm.values()

        b_offsets = torch.arange(B, device=device) * Fc
        new_rows = (b_offsets[:, None] + sm_idx[0][None, :]).reshape(-1)
        new_cols = sm_idx[1].unsqueeze(0).expand(B, -1).reshape(-1)
        new_vals = sm_val.unsqueeze(0).expand(B, -1).reshape(-1)
    else:
        # Batched 3D sparse (B, F*c, M): flatten batch into rows
        M = sparse_matrix.shape[2]
        sm = sparse_matrix.coalesce()
        sm_idx = sm.indices()  # (3, nnz) — [batch, row, col]
        sm_val = sm.values()

        new_rows = sm_idx[0] * Fc + sm_idx[1]
        new_cols = sm_idx[2]
        new_vals = sm_val

    sm_stacked = torch.sparse_coo_tensor(
        torch.stack([new_rows, new_cols]),
        new_vals,
        (B * Fc, M)
    )

    # 3. Single sparse @ sparse matmul
    result = (P @ sm_stacked).coalesce()

    # 4. Reshape (B*F*r, M) sparse → (B, F*r, M) batched sparse
    Fr = F * r
    res_idx = result.indices()
    res_val = result.values()

    return torch.sparse_coo_tensor(
        torch.stack([res_idx[0] // Fr, res_idx[0] % Fr, res_idx[1]]),
        res_val,
        (B, Fr, M)
    )

@torch.compile()
def local_basis(V, F, eps=1e-12):
    """
    PyTorch vectorized version of igl::local_basis.

    Args:
        V: (B, N, 3) or (N, 3) vertex positions
        F: (B, F, 3) or (F, 3) face indices (int/long)

    Returns:
        B1, B2, B3: each (B, F, 3) or (F, 3) local basis vectors per face
    """
    batched = V.dim() == 3
    if not batched:
        V = V.unsqueeze(0)
        F = F.unsqueeze(0)

    # Gather triangle vertices: (B, F, 3)
    v0 = torch.gather(V, 1, F[:, :, 0:1].expand(-1, -1, 3))
    v1 = torch.gather(V, 1, F[:, :, 1:2].expand(-1, -1, 3))
    v2 = torch.gather(V, 1, F[:, :, 2:3].expand(-1, -1, 3))

    e1 = v1 - v0
    B1 = e1 / (e1.norm(dim=-1, keepdim=True) + eps)

    t = v2 - v0
    B3 = torch.cross(B1, t, dim=-1)
    B3 = B3 / (B3.norm(dim=-1, keepdim=True) + eps)

    B2 = torch.cross(B1, B3, dim=-1)
    B2 = B2 / (B2.norm(dim=-1, keepdim=True) + eps)

    if not batched:
        B1, B2, B3 = B1.squeeze(0), B2.squeeze(0), B3.squeeze(0)
    return torch.stack([B1, -B2, B3], dim=-1)

@torch.no_grad()
def construct_mesh_operators(V, F, high_precision=True, three_D=False, get_both=False, get_extra=False):
    """ Returns mesh operators (Adapted from https://github.com/ArmanMaesumi/poissonnet.)

    Args:
        V: vertices (b, V, 3)
        F: faces (b, F, 3)
        high_precision: use double for intermediate computations
        three_D: whether to return three dimensional gradient operator and face areas

    Returns:
        vertex_mass:  (b, V)      lumped vertex masses
        solver:       [b,]        list of Cholesky solvers for mesh's cotangent Laplacian
        G:            (b, 2F, V) or (b, 3F, V)  face-based intrinsic / extrinsic gradient operator
        M:            (b, 2F) or (b, 3F)  interleaved face areas
        B:            (b, F, 3, 3) triangle bases

    """
    # ensure contiguous memory layout before calling CUDA extension
    V = V.contiguous()
    F = F.contiguous()
    if V.ndim == 2:
        V = V.unsqueeze(0)
        F = F.unsqueeze(0)
            
    if high_precision:
        V = V.double()

    nB, nV, _ = V.shape
    device = V.device

    vert_mass       = TMO.vertex_mass_batched(V, F, 1e-8)                     # (b, v)
    face_areas      = TMO.face_areas_batched(V, F)                            # (b, f)
    edge_lengths    = TMO.edge_lengths_batched(V, F)                          # (b, f, 3)
    G               = TMO.intrinsic_gradient_batched(edge_lengths, F)         # (b, 2f, v)
    B               = local_basis(V, F)
    if three_D:
        G = apply_sparse_rotation(G, B[...,:2])

    vert_mass = vert_mass.float()
    face_areas = face_areas.float()
    G = G.float()
    B = B.float()
    M = torch.repeat_interleave(face_areas, 3 if three_D else 2, dim=-1) # grad operator is interleaved convention, so M is interleaved too

    solvers = []
    Ls = TMO.cotangent_laplacian_batched(V, F, 1e-10).float() # (b, v, v)

    # Create Cholesky solver for each Laplacian in batch
    # This could be block-diagonalized, but speedup seems marginal
    for bi in range(nB):
        L = Ls[bi]
        eps = 1e-6
        sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
        L = L + sparse_eps_diag # (b, v, v)
        
        nretry = 0
        while nretry < 5:
            try:
                nrows = L.shape[-1]
                ii = L._indices()[0]
                jj = L._indices()[1]
                x = L._values()
                solver = cholespy.CholeskySolverF(
                    n_rows=nrows, ii=ii, jj=jj,
                    x=x, type=cholespy.MatrixType.COO
                )
                solvers.append(solver)
                break
            except Exception as e:
                eps = eps * 10.0
                sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
                L = L + sparse_eps_diag
                nretry += 1
        if nretry >= 5:
            raise RuntimeError(f"Failed to create Cholesky solver for mesh {bi} in batch")
    if get_both:
        G3 = apply_sparse_rotation(G, B[...,:2])
        M3 = torch.repeat_interleave(face_areas, 3, dim=-1)
        return vert_mass, solvers, G, G3, M, M3, B
    if get_extra:
        return vert_mass, solvers, G, M, B, face_areas, Ls
    return vert_mass, solvers, G, M, B
