"""
run_linear_hyperbench.py
─────────────────────────────────────────────────────────────────────────────
Évalue UNE variante d'optimisation du modèle linéaire G-SCOTT-Tucker sur le
protocole HyperBench Pavia identique au run ALS historique (mode fast) :
PSF gaussienne/Airy/sinc × ratios ×4/×8, SNR 35/30 dB, mêmes graines.

Usage : python run_linear_hyperbench.py <adam_prox|adam_l1|sgd_prox> [iters]

Produit :
    results_linear_<méthode>_fast.csv    métriques par cas
    results_linear_<méthode>_hist.json   courbes de convergence par cas
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from hyperbench import (BaseAdapter, ReconstructionInputs,
                        BenchmarkConfig, DegradationSpec, run_benchmark)
from linear_tucker_torch import fuse_linear_torch, METHODS

SCENE_PATH = 'pavia.mat'
SCENE_KEY = 'paviaU'


# ── Q2n hypercomplexe (Garzelli & Nencini 2009) — implémentation standalone ─
def _q2n_conj(x):
    """Conjugaison hypercomplexe sur le dernier axe : [x0, -x1, ..., -xn]."""
    return np.concatenate([x[..., :1], -x[..., 1:]], -1)


def _q2n_onion_mult(o1, o2):
    """Produit 'onion' hypercomplexe, vectorisé sur toutes les dims de tête.
    Équivalent aux _om1D/_om2D récursifs, calculé UNE fois pour toutes les
    bandes et tous les blocs (l'ancienne version le recalculait par bande)."""
    n = o1.shape[-1]
    if n == 1:
        return o1 * o2
    L = n // 2
    a, b = o1[..., :L], _q2n_conj(o1[..., L:])
    c, d = o2[..., :L], _q2n_conj(o2[..., L:])
    return np.concatenate(
        [_q2n_onion_mult(a, c) - _q2n_onion_mult(d, _q2n_conj(b)),
         _q2n_onion_mult(_q2n_conj(a), d) + _q2n_onion_mult(c, b)], -1)


def _q2n_blocks(d1, d2, ws):
    """Vecteurs Q par bloc — d1, d2 : (B, ws, ws, N). Vectorisé sur B."""
    M = ws * ws
    f = M / (M - 1)
    d1 = d1.astype(np.float64).copy()
    d2 = _q2n_conj(d2.astype(np.float64))
    # Normalisation par les statistiques de la référence (par bloc et bande)
    s = d1.mean(axis=(1, 2), keepdims=True)                 # (B,1,1,N)
    t = d1.std(axis=(1, 2), ddof=1, keepdims=True)
    t = np.where(t == 0, np.finfo(float).eps, t)
    d1 = (d1 - s) / t + 1.0
    s_nz = (s != 0)
    d2_b0 = np.where(s_nz, (d2 - s) / t + 1.0, d2 - s + 1.0)
    d2_bn = np.where(s_nz, -((-d2 - s) / t + 1.0), -(-d2 - s + 1.0))
    d2 = np.concatenate([d2_b0[..., :1], d2_bn[..., 1:]], -1)

    m1, m2 = d1.mean(axis=(1, 2)), d2.mean(axis=(1, 2))     # (B,N)
    mq1m = np.sqrt((m1 ** 2).sum(-1))                        # (B,)
    mq2m = np.sqrt((m2 ** 2).sum(-1))
    mq1 = np.sqrt((d1 ** 2).sum(-1))                         # (B,ws,ws)
    mq2 = np.sqrt((d2 ** 2).sum(-1))
    t2, t4 = mq1m * mq2m, mq1m ** 2 + mq2m ** 2
    t3 = f * ((mq1 ** 2).mean(axis=(1, 2)) + (mq2 ** 2).mean(axis=(1, 2))) - f * t4
    mb = np.where(t4 > 0, 2.0 * t2 / np.where(t4 > 0, t4, 1.0), 0.0)

    qv = f * _q2n_onion_mult(d1, d2).mean(axis=(1, 2))       # (B,N)
    qm = _q2n_onion_mult(m1, m2)                              # (B,N)
    t3_safe = np.where(t3 == 0, 1.0, t3)
    q = (qv - f * qm) * mb[:, None] * (2.0 / t3_safe[:, None])
    if (t3 == 0).any():                                       # cas dégénéré
        qz = np.zeros_like(q)
        qz[:, -1] = mb
        q = np.where((t3 == 0)[:, None], qz, q)
    return q


def _calc_q2n(ref, fused, ws=32):
    """Index Q2n hypercomplexe (Garzelli & Nencini 2009), NumPy vectorisé."""
    ref = ref.astype(np.float64); fused = fused.astype(np.float64)
    H, W, N3 = ref.shape; stride = ws
    stepx = max(1, int(np.ceil(H / stride))); stepy = max(1, int(np.ceil(W / stride)))
    est1 = (stepx - 1) * stride + ws - H; est2 = (stepy - 1) * stride + ws - W
    if est1 != 0 or est2 != 0:
        rp = np.zeros((H + est1, W + est2, N3)); fp = np.zeros_like(rp)
        rp[:H, :W] = ref; fp[:H, :W] = fused
        if est2 > 0: rp[:H, W:] = ref[:, W-est2:W][:, ::-1]; fp[:H, W:] = fused[:, W-est2:W][:, ::-1]
        if est1 > 0: rp[H:] = rp[H-est1:H][::-1]; fp[H:] = fp[H-est1:H][::-1]
        ref, fused = rp, fp; H, W = ref.shape[:2]
    n2 = 1
    while n2 < N3: n2 *= 2
    if n2 > N3:
        pad = np.zeros((H, W, n2 - N3)); ref = np.concatenate([ref, pad], 2); fused = np.concatenate([fused, pad], 2)
    N3 = n2

    # Découpage en blocs non chevauchants : (stepx·stepy, ws, ws, N3)
    def blocks(x):
        return (x.reshape(stepx, ws, stepy, ws, N3)
                 .transpose(0, 2, 1, 3, 4).reshape(-1, ws, ws, N3))

    q = _q2n_blocks(blocks(ref), blocks(fused), ws)          # (B, N3)
    return float(np.mean(np.sqrt((q ** 2).sum(-1))))


def load_gt_scene(scene_path, scene_key):
    """Vérité terrain avec la MÊME normalisation que HyperBench :
    clipping percentile 1–99 puis mise à l'échelle [0,1] (cf. normalize_image)."""
    import scipy.io as sio
    _d = sio.loadmat(scene_path)
    S_raw = _d[scene_key].astype(np.float64)
    lo, hi = np.percentile(S_raw, [1.0, 99.0])
    denom = max(float(hi - lo), 1e-12)
    return (np.clip(S_raw, lo, hi) - lo) / denom


class LinearTuckerAdapter(BaseAdapter):
    """Tucker couplé linéaire, résolu par gradient (variante paramétrable)."""
    def __init__(self, method, ranks=(15, 15, 8), beta=1e-6, iters=3000, lr=1e-2,
                 scheduler_type='constant', patience=10000, tol_es=1e-7, prox_step=None,
                 scene_path=None, scene_key=None):
        super().__init__(name=f'Tucker-lin ({method})', shape_policy='crop')
        self.method, self.ranks, self.beta = method, ranks, beta
        self.iters, self.lr = iters, lr
        self.scheduler_type = scheduler_type
        self.patience = patience
        self.tol_es = tol_es
        self.prox_step = prox_step
        self.histories = []           # une entrée par cas, dans l'ordre
        # Vérité terrain (normalisation HyperBench) pour le calcul Q2n
        try:
            self._gt_scene = load_gt_scene(scene_path or SCENE_PATH,
                                           scene_key or SCENE_KEY)
        except Exception:
            self._gt_scene = None

    def predict(self, inputs: ReconstructionInputs):
        S, stats = fuse_linear_torch(
            hr_msi=np.array(inputs.hr_msi, dtype=np.float64),
            lr_hsi=np.array(inputs.lr_hsi, dtype=np.float64),
            srf=np.array(inputs.srf, dtype=np.float64),
            psf_kernel=np.array(inputs.psf, dtype=np.float64),
            method=self.method, ranks=self.ranks, beta=self.beta,
            iters=self.iters, lr=self.lr,
            scheduler_type=self.scheduler_type, patience=self.patience,
            tol_es=self.tol_es, prox_step=self.prox_step)
        self.histories.append(stats.pop('history'))
        # ── Q2n hypercomplexe ───────────────────────────────────────────────
        try:
            meta = inputs.metadata or {}
            gt_shape = tuple(meta.get('gt_shape', ()))  # (H, W, C) après crop
            S_q = np.clip(S, 0.0, 1.0)
            if self._gt_scene is not None and len(gt_shape) == 3:
                H, W, C = gt_shape
                # La politique 'crop' prend le coin haut-gauche
                gt_crop = self._gt_scene[:H, :W, :C]
                stats['q2n'] = _calc_q2n(gt_crop, S_q)
            elif self._gt_scene is not None:
                H, W = S_q.shape[:2]
                gt_crop = self._gt_scene[:H, :W, :S_q.shape[2]]
                stats['q2n'] = _calc_q2n(gt_crop, S_q)
            else:
                stats['q2n'] = float('nan')
        except Exception as _e:
            stats['q2n'] = float('nan')
        return S, stats



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="run_linear_hyperbench")
    parser.add_argument('positional_method', nargs='?', default=None)
    parser.add_argument('positional_iters', type=int, nargs='?', default=None)
    
    parser.add_argument('--method', default='adam_prox')
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--beta', type=float, default=1e-6)
    parser.add_argument('--ranks', default='15,15,8')
    parser.add_argument('--scheduler', default='constant')
    parser.add_argument('--patience', type=int, default=10000)
    parser.add_argument('--tol_es', type=float, default=1e-7)
    parser.add_argument('--prox_step', type=float, default=None)
    parser.add_argument('--tag', default=None)

    args = parser.parse_args()

    method = args.positional_method if args.positional_method is not None else args.method
    assert method in METHODS, f'méthode inconnue : {method} (choix : {METHODS})'
    iters = args.positional_iters if args.positional_iters is not None else args.iters
    lr = args.lr
    beta = args.beta
    ranks = tuple(map(int, args.ranks.split(',')))
    scheduler = args.scheduler
    patience = args.patience
    tol_es = args.tol_es
    prox_step = args.prox_step

    tag = args.tag if args.tag is not None else method

    cfg = BenchmarkConfig(
        scene_path=SCENE_PATH, scene_key=SCENE_KEY,
        psf_names=['gaussian', 'airy', 'sinc'],
        psf_sigmas=[3.4], psf_kernel_radii=[7],
        degradation_specs=[
            DegradationSpec(4, 4, 35., 40.),
            DegradationSpec(8, 4, 30., 40.),
        ],
        metrics=['psnr', 'sam', 'ergas', 'ssim'],
        save_csv=True,
        output_csv_path=f'results_linear_{tag}_fast.csv',
        overwrite_csv_on_start=True, fail_fast=False,
    )

    print(f'=== Tucker linéaire [{method}] x HyperBench — tag={tag} ===')
    adapter = LinearTuckerAdapter(method, ranks=ranks, beta=beta, iters=iters, lr=lr,
                                 scheduler_type=scheduler, patience=patience,
                                 tol_es=tol_es, prox_step=prox_step)
    run_benchmark(adapter, cfg)

    with open(f'results_linear_{tag}_hist.json', 'w') as f:
        json.dump(adapter.histories, f)
    print(f'ok : results_linear_{tag}_fast.csv + historiques ({len(adapter.histories)} cas)')
