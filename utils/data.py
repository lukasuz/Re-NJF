import os
import glob
import random
import multiprocessing as mp
import joblib

from omegaconf import OmegaConf
import roma
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from utils.mesh import norm_pair

def load_config(path="configs/default.yaml", overrides=None):
    config = OmegaConf.load(path)
    if "inherits" in config:
        parent_path = config.inherits
        if not os.path.isabs(parent_path):
            parent_path = os.path.join(os.path.dirname(path), parent_path)
        parent_config = load_config(parent_path)
        del config["inherits"]
        config = OmegaConf.merge(parent_config, config)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config
    

def rotvec_to_euler(thetas: torch.Tensor, convention: str = 'xyz') -> torch.Tensor:
    """Convert pose parameters (rotvec) to Euler angles."""
    leading = thetas.shape[:-1]
    thetas_flat = thetas.reshape(-1, 3)
    rotmats = roma.rotvec_to_rotmat(thetas_flat)
    euler = Rotation.from_matrix(
        rotmats.cpu().detach().numpy()
    ).as_euler(convention)
    euler = torch.from_numpy(euler.astype(np.float32)).to(thetas.device)
    return euler.reshape(*leading, -1, 3)


def load_features_from_dir(dirpath):
    """Load all .pt feature files from dirpath, sorted numerically. Returns (N, V, C) tensor."""
    pt_files = sorted(
        glob.glob(os.path.join(dirpath, '**', '*.pt'), recursive=True),
        key=lambda f: int(''.join(filter(str.isdigit, os.path.splitext(os.path.basename(f))[0]))),
    )
    return torch.stack([torch.load(f, map_location='cpu', weights_only=True) for f in pt_files], dim=0)


def compute_relative_rotvec(src_rv: torch.Tensor, tar_rv: torch.Tensor) -> torch.Tensor:
    """Relative Rodriguez rotation from src to tar per joint: R_src^{-1} @ R_tar."""
    B, D = src_rv.shape
    N = D // 3
    R_src = roma.rotvec_to_rotmat(src_rv.reshape(B * N, 3))
    R_tar = roma.rotvec_to_rotmat(tar_rv.reshape(B * N, 3))
    R_rel = R_src.mT @ R_tar
    return roma.rotmat_to_rotvec(R_rel).reshape(B, D)


@torch.no_grad()
def generate_pose_pairs(
    vp, num_pairs, body_shape_std, pose_std, latent_dim, device, seed=None, gen=None,
    num_betas=10, randomize_hands=False, **kwargs
):
    """Generate (src, target) pose-shape pairs reproducibly."""
    if gen is None:
        gen = torch.Generator()
        gen.manual_seed(seed)

    betas = torch.randn(num_pairs, num_betas, generator=gen) * body_shape_std
    genders = torch.randint(0, 2, (num_pairs,), generator=gen)

    src_z = pose_std * torch.randn(num_pairs, latent_dim, generator=gen)
    src_poses_63 = vp.decode(src_z.to(device))['pose_body'].detach().cpu()

    tar_z = pose_std * torch.randn(num_pairs, latent_dim, generator=gen)
    tar_poses_63 = vp.decode(tar_z.to(device))['pose_body'].detach().cpu()

    hand_std = 1 / 3
    global_orient = torch.zeros(num_pairs, 3)
    if randomize_hands:
        src_hands = hand_std * torch.randn(num_pairs, 6, generator=gen)
        tar_hands = hand_std * torch.randn(num_pairs, 6, generator=gen)
    else:
        src_hands = torch.zeros(num_pairs, 6)
        tar_hands = torch.zeros(num_pairs, 6)
    src_poses = torch.cat([global_orient, src_poses_63, src_hands], dim=-1)
    tar_poses = torch.cat([global_orient, tar_poses_63, tar_hands], dim=-1)

    return dict(betas=betas, genders=genders,
                src_z=src_z, src_poses=src_poses,
                tar_z=tar_z, tar_poses=tar_poses)


def create_optimizer(parameters, config):
    """Create AdamW optimizer and ExponentialLR scheduler."""
    optimizer = torch.optim.AdamW(parameters, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.scheduler_gamma
    )
    return optimizer, scheduler


def _load_worker(fold, queue, amass_path):
    data = joblib.load(os.path.join(amass_path, f'amass_30fps_{fold}.pt'))
    thetas = np.concat(data['thetas'])[:, 3:]
    queue.put(thetas)


def extract_thetas(fold, amass_path):
    q = mp.Queue()
    p = mp.Process(target=_load_worker, args=(fold, q, amass_path))
    p.start()
    thetas = q.get()
    p.join()
    return torch.tensor(thetas, dtype=torch.float32)


def get_test_sequences(amass_path):
    data = joblib.load(os.path.join(amass_path, 'amass_30fps_test.pt'))
    return [torch.tensor(thetas[:, 3:], dtype=torch.float32) for thetas in data['thetas']]


# ---------------------------------------------------------------------------
# Dataset / iterator classes
# ---------------------------------------------------------------------------

class AMASSDataIterator:
    """Infinite training source + finite val iterator using AMASS poses with STAR meshes."""

    def __init__(self, config, device, batch_size, load_features: bool = True):
        from star.pytorch.star import STAR

        self.device = device
        self.batch_size = batch_size
        self.feature_dim = 256
        self.pose_ndim = 69

        num_betas = config.get('num_betas', 10)
        body_shape_std = config.get('body_shape_std', 2.0)
        seed = config.get('seed', 42)
        self._num_betas = num_betas
        self._body_shape_std = body_shape_std

        star_path = config['star_path']
        self._star_models = [
            STAR(os.path.join(star_path, 'male/model.npz'), num_betas=num_betas),
            STAR(os.path.join(star_path, 'female/model.npz'), num_betas=num_betas),
        ]
        self._num_verts = 6890
        with torch.no_grad():
            _d = self._star_models[0].forward(
                torch.zeros(1, 72).to(device),
                torch.zeros(1, num_betas).to(device),
                torch.zeros(1, 3).to(device),
            )
            self.faces = torch.from_numpy(_d.f.astype(np.int64)).to(device)

        amass_path = config['amass_path']
        print("Loading AMASS thetas (train)...")
        train_thetas = extract_thetas('train', amass_path)
        print("Loading AMASS thetas (vald)...")
        val_thetas = extract_thetas('vald', amass_path)
        print("Loading AMASS thetas (test)...")
        test_thetas = extract_thetas('test', amass_path)
        self._train_thetas = train_thetas

        self._train_gen = torch.Generator().manual_seed(seed)

        num_val = config.get('amass_num_val_pairs', 500)
        vg = torch.Generator().manual_seed(seed + 1)
        M = len(val_thetas)
        self._val_src = val_thetas[torch.randint(0, M, (num_val,), generator=vg)]
        self._val_tar = val_thetas[torch.randint(0, M, (num_val,), generator=vg)]
        self._val_betas = torch.randn(num_val, num_betas, generator=vg) * body_shape_std
        self._val_genders = torch.randint(0, 2, (num_val,), generator=vg).tolist()
        self.num_val = num_val

        num_test = config.get('amass_num_test_pairs', 1000)
        tg = torch.Generator().manual_seed(seed + 2)
        K = len(test_thetas)
        self._test_src = test_thetas[torch.randint(0, K, (num_test,), generator=tg)]
        self._test_tar = test_thetas[torch.randint(0, K, (num_test,), generator=tg)]
        self._test_betas = torch.randn(num_test, num_betas, generator=tg) * body_shape_std
        self._test_genders = torch.randint(0, 2, (num_test,), generator=tg).tolist()
        self.num_test = num_test

        if load_features:
            self._train_feats = load_features_from_dir(config['amass_train_features_path'])
            self._val_feats = load_features_from_dir(config['amass_val_features_path'])
            self._test_feats = load_features_from_dir(config['amass_test_features_path'])

        self._use_relative_pose = config.get('use_relative_pose', False)
        self._use_canonical = config.get('use_canonical', False)

        feat_info = f", feature pool train={self._train_feats.shape[0]}" if load_features else ""
        print(f"AMASS: {len(train_thetas)} train / {len(val_thetas)} val / {len(test_thetas)} test thetas{feat_info}")

    def __iter__(self):
        return self

    @torch.no_grad()
    def __next__(self):
        N, B = self._train_thetas.shape[0], self.batch_size
        src = self._train_thetas[torch.randint(0, N, (B,), generator=self._train_gen)]
        tar = self._train_thetas[torch.randint(0, N, (B,), generator=self._train_gen)]
        if self._use_canonical:
            src = torch.zeros_like(src)
        betas = torch.randn(B, self._num_betas, generator=self._train_gen) * self._body_shape_std
        genders = torch.randint(0, 2, (B,), generator=self._train_gen).tolist()

        go = torch.zeros(B, 3)
        src_v = self._generate_star_meshes(betas, torch.cat([go, src], -1), genders)
        tar_v = self._generate_star_meshes(betas, torch.cat([go, tar], -1), genders)
        faces_b = self.faces.unsqueeze(0).expand(B, -1, -1)

        if hasattr(self, '_train_feats'):
            fi = torch.randint(0, self._train_feats.shape[0], (B,), generator=self._train_gen)
            vertex_features = self._train_feats[fi].float().to(self.device)
        else:
            vertex_features = None

        if self._use_relative_pose:
            cond_feat = compute_relative_rotvec(src.to(self.device), tar.to(self.device))
        else:
            cond_feat = tar.to(self.device)

        src_v, tar_v = norm_pair(src_v, tar_v)
        return src_v, tar_v, faces_b, cond_feat, vertex_features

    @torch.no_grad()
    def _generate_star_meshes(self, betas, poses_72, genders):
        B = betas.shape[0]
        verts = torch.empty(B, self._num_verts, 3, device=self.device)
        for gi, star in enumerate(self._star_models):
            mask = [i for i in range(B) if genders[i] == gi]
            if not mask:
                continue
            out = star.forward(
                poses_72[mask].to(self.device),
                betas[mask].to(self.device),
                torch.zeros(len(mask), 3, device=self.device),
            )
            verts[mask] = out.detach()
        return verts

    def _iter_pairs(self, src_thetas, tar_thetas, betas, genders, feats):
        N = len(betas)
        for s in range(0, N, self.batch_size):
            e = min(s + self.batch_size, N)
            B_v = e - s
            go = torch.zeros(B_v, 3)
            src_slice = torch.zeros_like(src_thetas[s:e]) if self._use_canonical else src_thetas[s:e]
            src_v = self._generate_star_meshes(betas[s:e], torch.cat([go, src_slice], -1), genders[s:e])
            tar_v = self._generate_star_meshes(betas[s:e], torch.cat([go, tar_thetas[s:e]], -1), genders[s:e])
            faces_b = self.faces.unsqueeze(0).expand(B_v, -1, -1)
            if feats is not None:
                fi = torch.tensor([(s + j) % feats.shape[0] for j in range(B_v)])
                vertex_features = feats[fi].float().to(self.device)
            else:
                vertex_features = None
            cond_feat = tar_thetas[s:e].to(self.device)
            if self._use_relative_pose:
                cond_feat = compute_relative_rotvec(src_thetas[s:e].to(self.device), cond_feat)
            src_v, tar_v = norm_pair(src_v, tar_v)
            yield src_v, tar_v, faces_b, cond_feat, vertex_features

    @torch.no_grad()
    def val_iter(self):
        yield from self._iter_pairs(
            self._val_src, self._val_tar, self._val_betas, self._val_genders,
            self._val_feats if hasattr(self, '_val_feats') else None,
        )

    @torch.no_grad()
    def test_iter(self):
        yield from self._iter_pairs(
            self._test_src, self._test_tar, self._test_betas, self._test_genders,
            self._test_feats if hasattr(self, '_test_feats') else None,
        )


class SMALDataIterator:
    """Infinite training source + finite val iterator using SMAL animal model."""

    def __init__(self, config, device, batch_size, load_features: bool = True):
        import json
        from SMALify import SMAL

        self.device = device
        self.batch_size = batch_size
        self.feature_dim = 256
        self.pose_ndim = 105

        smal_shape_std = config.get('smal_shape_std', 2.0)
        seed = config.get('seed', 42)
        self._num_betas = 20
        self._body_shape_std = smal_shape_std

        animal3d_path = config['animal3d_path']
        with open(os.path.join(animal3d_path, 'train.json')) as f:
            train_data = json.load(f)['data']
        with open(os.path.join(animal3d_path, 'test.json')) as f:
            test_data = json.load(f)['data']

        self._train_poses = torch.tensor(
            [e['pose'] for e in train_data], dtype=torch.float32)
        self._train_poses[:, :3] = 0.0
        self._test_poses = torch.tensor(
            [e['pose'] for e in test_data], dtype=torch.float32)
        self._test_poses[:, :3] = 0.0

        N_train = len(self._train_poses)
        num_val = config.get('smal_num_val_pairs', 100)
        val_indices = torch.linspace(0, N_train - 1, num_val).long()
        val_poses_pool = self._train_poses[val_indices]

        self._smal_models = [SMAL(device, fid) for fid in range(5)]
        self._num_verts = 3889
        self.faces = self._smal_models[0].faces.long().to(device)

        self._train_gen = torch.Generator().manual_seed(seed)

        vg = torch.Generator().manual_seed(seed + 1)
        M_val = len(val_poses_pool)
        self._val_src = val_poses_pool[torch.randint(0, M_val, (num_val,), generator=vg)]
        self._val_tar = val_poses_pool[torch.randint(0, M_val, (num_val,), generator=vg)]
        self._val_betas = torch.randn(num_val, self._num_betas, generator=vg) * smal_shape_std
        self._val_fids = torch.randint(0, 5, (num_val,), generator=vg).tolist()
        self.num_val = num_val

        num_test = config.get('smal_num_test_pairs', 100)
        tg = torch.Generator().manual_seed(seed + 2)
        M_test = len(self._test_poses)
        self._test_src = self._test_poses[torch.randint(0, M_test, (num_test,), generator=tg)]
        self._test_tar = self._test_poses[torch.randint(0, M_test, (num_test,), generator=tg)]
        self._test_betas = torch.randn(num_test, self._num_betas, generator=tg) * smal_shape_std
        self._test_fids = torch.randint(0, 5, (num_test,), generator=tg).tolist()
        self.num_test = num_test

        if load_features:
            self._train_feats = load_features_from_dir(config['smal_train_features_path'])
            self._val_feats = load_features_from_dir(config['smal_val_features_path'])
            self._test_feats = load_features_from_dir(config['smal_test_features_path'])

        self._use_relative_pose = config.get('use_relative_pose', False)
        self._random_smal_pose = config.get('random_smal_pose', False)
        self._use_canonical = config.get('use_canonical', False)

        angles = rotvec_to_euler(self._train_poses)
        self._euler_mean = angles.mean(dim=0)
        self._euler_std = angles.std(dim=0)
        self._euler_amax = angles.max(dim=0)[0]
        self._euler_amin = angles.min(dim=0)[0]

        feat_info = f", feature pool train={self._train_feats.shape[0]}" if load_features else ""
        print(f"SMAL: {N_train} train / {len(self._test_poses)} test poses, "
              f"{num_val} fixed val / {num_test} fixed test pairs{feat_info}")

    def __iter__(self):
        return self

    def _sample_random_poses(self, B: int) -> torch.Tensor:
        from scipy.spatial.transform import Rotation
        euler = torch.randn(B, 35, 3) * self._euler_std + self._euler_mean
        euler = torch.clamp(euler, self._euler_amin, self._euler_amax)
        R = Rotation.from_euler('xyz', euler.reshape(-1, 3).numpy()).as_matrix()
        R = torch.from_numpy(R.astype(np.float32))
        rotvec = roma.rotmat_to_rotvec(R).reshape(B, 105)
        rotvec[:, :3] = 0.0
        return rotvec

    @torch.no_grad()
    def __next__(self):
        N, B = len(self._train_poses), self.batch_size
        if self._random_smal_pose:
            src = self._sample_random_poses(B)
            tar = self._sample_random_poses(B)
        else:
            src = self._train_poses[torch.randint(0, N, (B,), generator=self._train_gen)]
            tar = self._train_poses[torch.randint(0, N, (B,), generator=self._train_gen)]
        if self._use_canonical:
            src = torch.zeros_like(src)
        betas = torch.randn(B, self._num_betas, generator=self._train_gen) * self._body_shape_std
        fids = torch.randint(0, 5, (B,), generator=self._train_gen).tolist()

        src_v = self._generate_smal_meshes(betas, src, fids)
        tar_v = self._generate_smal_meshes(betas, tar, fids)
        faces_b = self.faces.unsqueeze(0).expand(B, -1, -1)

        if hasattr(self, '_train_feats'):
            fi = torch.randint(0, self._train_feats.shape[0], (B,), generator=self._train_gen)
            vertex_features = self._train_feats[fi].float().to(self.device)
        else:
            vertex_features = None

        if self._use_relative_pose:
            cond_feat = compute_relative_rotvec(src.to(self.device), tar.to(self.device))
        else:
            cond_feat = tar.to(self.device)

        src_v, tar_v = norm_pair(src_v, tar_v)
        return src_v, tar_v, faces_b, cond_feat, vertex_features

    @torch.no_grad()
    def _generate_smal_meshes(self, betas, poses_105, shape_family_ids):
        B = betas.shape[0]
        verts = torch.empty(B, self._num_verts, 3, device=self.device)
        poses_reshaped = poses_105.reshape(B, 35, 3)
        for fid in range(5):
            mask = [i for i in range(B) if shape_family_ids[i] == fid]
            if not mask:
                continue
            out = self._smal_models[fid](
                poses_reshaped[mask].to(self.device),
                betas[mask].to(self.device),
            )
            verts[mask] = out[0].detach()
        R = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], device=self.device)
        return verts @ R.T

    def _iter_pairs(self, src_poses, tar_poses, betas, fids, feats):
        N = len(betas)
        for s in range(0, N, self.batch_size):
            e = min(s + self.batch_size, N)
            B_v = e - s
            src_slice = torch.zeros_like(src_poses[s:e]) if self._use_canonical else src_poses[s:e]
            src_v = self._generate_smal_meshes(betas[s:e], src_slice, fids[s:e])
            tar_v = self._generate_smal_meshes(betas[s:e], tar_poses[s:e], fids[s:e])
            faces_b = self.faces.unsqueeze(0).expand(B_v, -1, -1)
            if feats is not None:
                fi = torch.tensor([(s + j) % feats.shape[0] for j in range(B_v)])
                vertex_features = feats[fi].float().to(self.device)
            else:
                vertex_features = None
            cond_feat = tar_poses[s:e].to(self.device)
            if self._use_relative_pose:
                cond_feat = compute_relative_rotvec(src_poses[s:e].to(self.device), cond_feat)
            src_v, tar_v = norm_pair(src_v, tar_v)
            yield src_v, tar_v, faces_b, cond_feat, vertex_features

    @torch.no_grad()
    def _generate_balanced(self, poses_pool, n, seed):
        n_per = n // 5
        N = poses_pool.shape[0]
        g = torch.Generator().manual_seed(seed)
        all_verts, all_fids = [], []
        for fid in range(5):
            idx = torch.randint(0, N, (n_per,), generator=g)
            poses = poses_pool[idx]
            betas = torch.randn(n_per, self._num_betas, generator=g) * self._body_shape_std
            verts = self._generate_smal_meshes(betas, poses, [fid] * n_per)
            all_verts.append(verts.cpu())
            all_fids.extend([fid] * n_per)
        return torch.cat(all_verts, dim=0).numpy(), all_fids

    def generate_train_balanced(self, n):
        return self._generate_balanced(self._train_poses, n, seed=0)

    def generate_val_balanced(self, n):
        return self._generate_balanced(self._train_poses, n, seed=1)

    def generate_test_balanced(self, n):
        return self._generate_balanced(self._test_poses, n, seed=2)

    @torch.no_grad()
    def val_iter(self):
        yield from self._iter_pairs(
            self._val_src, self._val_tar, self._val_betas, self._val_fids,
            self._val_feats if hasattr(self, '_val_feats') else None,
        )

    @torch.no_grad()
    def test_iter(self):
        yield from self._iter_pairs(
            self._test_src, self._test_tar, self._test_betas, self._test_fids,
            self._test_feats if hasattr(self, '_test_feats') else None,
        )


class AMASSSMALDataIterator:
    """Wraps AMASSDataIterator and SMALDataIterator into one combined iterator."""

    def __init__(self, config, device, batch_size, load_features: bool = True):
        self.amass_iter = AMASSDataIterator(config, device, batch_size, load_features=load_features)
        self.smal_iter = SMALDataIterator(config, device, batch_size, load_features=load_features)

        n_amass = len(self.amass_iter._train_thetas)
        n_smal = len(self.smal_iter._train_poses)
        if config.get('random_smal_pose', False):
            self._p_amass = 0.7
        else:
            self._p_amass = 0.95

        self.pose_ndim = (self.amass_iter.pose_ndim, self.smal_iter.pose_ndim)
        self.feature_dim = 256
        self.num_val = self.amass_iter.num_val + self.smal_iter.num_val
        self.num_test = self.amass_iter.num_test + self.smal_iter.num_test

        print(f"AMASSSMALDataIterator: p_amass={self._p_amass:.3f} "
              f"({n_amass} AMASS + {n_smal} SMAL train poses), "
              f"num_val={self.num_val}, num_test={self.num_test}")

    def __iter__(self):
        return self

    def __next__(self):
        if random.random() < self._p_amass:
            return next(self.amass_iter)
        return next(self.smal_iter)

    def val_iter(self):
        yield from self.amass_iter.val_iter()
        yield from self.smal_iter.val_iter()

    def test_iter(self):
        yield from self.amass_iter.test_iter()
        yield from self.smal_iter.test_iter()


def build_data_iterator(config, device, batch_size, load_features: bool = True):
    """Return the appropriate data iterator based on config flags."""
    if config.get('use_amass_smal', False):
        return AMASSSMALDataIterator(config, device, batch_size, load_features=load_features)
    if config.get('use_smal', False):
        return SMALDataIterator(config, device, batch_size, load_features=load_features)
    if config.get('use_amass', False):
        return AMASSDataIterator(config, device, batch_size, load_features=load_features)


def load_network(w_path, cfg, device, cfg_path=None):
    """Load a CustomPoissonNetv2 from checkpoint, inferring extra_features from weights.

    Returns (model, use_amass_smal): model is in eval mode; use_amass_smal indicates dual MLP.
    """
    from networks import CustomPoissonNetv2

    if w_path is None:
        assert cfg_path is not None, "cfg_path required to resolve default weights path"
        w_path = os.path.join(*cfg_path.split('/')[:-1], 'model_latest.pt')

    ckpt = torch.load(w_path, weights_only=True, map_location='cpu')
    C_width = cfg.pn_width
    use_amass_smal = 'extra_feat_mlp_smal.0.weight' in ckpt
    if use_amass_smal:
        amass_dim = ckpt['extra_feat_mlp.0.weight'].shape[1] - C_width
        smal_dim  = ckpt['extra_feat_mlp_smal.0.weight'].shape[1] - C_width
        extra_features = (amass_dim, smal_dim)
        print(f"[{cfg_path}] amass_smal: SMPL dim={amass_dim}, SMAL dim={smal_dim}")
    elif 'extra_feat_mlp.0.weight' in ckpt:
        extra_features = ckpt['extra_feat_mlp.0.weight'].shape[1] - C_width
        print(f"[{cfg_path}] inferred extra_features={extra_features}")
    else:
        extra_features = 0
        print(f"[{cfg_path}] fallback extra_features=0")

    use_xyz = cfg.get('use_xyz', False)
    m = CustomPoissonNetv2(
        C_in=3 if use_xyz else cfg.feature_dim,
        C_out=3,
        C_width=cfg.pn_width,
        n_blocks=cfg.pn_nblocks,
        head=cfg.get('pn_head', 're_njf'),
        extra_features=extra_features,
        config=dict(cfg),
    ).to(device)
    m.load_state_dict(ckpt)
    m.eval()
    print(f"Model loaded from {w_path}")
    return m, use_amass_smal
