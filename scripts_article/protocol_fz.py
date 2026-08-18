import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -*- coding: utf-8 -*-
"""
protocol_fz.py
─────────────────────────────────────────────────────────────────────────────
Réplication Python EXACTE du protocole d'évaluation MATLAB de l'équipe
(processMultiperspectralData.m + degradation_spatiale_S2_20m.m) :

  - PaviaU redimensionné à 1500×1500 (bilinéaire), λ = linspace(400, 2500, 103)
  - S2_10 : 4 bandes (490/560/665/842 nm) à 1500²
  - S2_20 : 6 bandes (705/740/783/865/1610/2190) au ratio 2 (750²)
  - S2_60 : 3 bandes (443/940/1375) au ratio 6 (250²)
  - S3    : 21 bandes OLCI au ratio 30 (50²)
  - Référence : les 21 bandes à 1500²  —  AUCUN BRUIT, dégradation bilinéaire
  - Opérateurs supposés par les algorithmes : gaussienne 3×3 Toeplitz,
    σ = 0.5 / (2·(4 ln 2) / ratio²), décimation start_pos.

Évalue nos modèles linéaires (Tucker couplé sparse) en version MULTI-FLUX
dans ces conditions : ALS-Lipschitz (v0) et Adam+prox (torch, 4 sources).
Métriques : PSNR, SAM, Q2n, ERGAS — contre la référence 21 bandes.
─────────────────────────────────────────────────────────────────────────────
"""
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
from scipy.linalg import toeplitz
import torch

from run_linear_hyperbench import _calc_q2n     # Q2n vectorisé

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

S2_BANDS = np.array([443, 490, 560, 665, 705, 740, 783, 842, 865, 940, 1375, 1610, 2190])
S2_RES = np.array([60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20])
S3_BANDS = np.array([400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25,
                     708.75, 753.75, 761.25, 764.375, 767.5, 778.75, 865, 885,
                     900, 940, 1020])


def bilinear_resize(img, out_hw):
    f0 = out_hw[0] / img.shape[0]
    f1 = out_hw[1] / img.shape[1]
    if img.ndim == 3:
        return zoom(img, (f0, f1, 1), order=1)
    return zoom(img, (f0, f1), order=1)


def simulate_streams(pavia_path='pavia.mat', key='paviaU', size=1500):
    """Réplique processMultiperspectralData.m (sans bruit, bilinéaire)."""
    cube = sio.loadmat(pavia_path)[key].astype(np.float64)
    cube = cube / cube.max()
    cube = bilinear_resize(cube, (size, size))
    wl = np.linspace(400, 2500, cube.shape[2])

    def nearest(bands):
        return np.array([np.argmin(np.abs(wl - b)) for b in bands])

    streams = {}
    for res, ratio in [(10, 1), (20, 2), (60, 6)]:
        idx = nearest(S2_BANDS[S2_RES == res])
        sub = cube[:, :, idx]
        streams[f's2_{res}'] = (bilinear_resize(sub, (size // ratio, size // ratio))
                                if ratio > 1 else sub)
    s3_full = cube[:, :, nearest(S3_BANDS)]
    streams['s3'] = bilinear_resize(s3_full, (size // 30, size // 30))
    ref = s3_full                                  # référence 21 bandes à 1500²
    return streams, ref


# ── Opérateurs supposés (port fidèle de degradation_spatiale_S2_20m.m) ────────

def fz_sigma(ratio):
    return (1.0 / (2.0 * 2.7725887 / ratio ** 2)) * 0.5


def gaussian_kernel2d(k, sig):
    ax = np.arange(k) - (k - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2.0 * sig ** 2))
    ker = np.outer(g, g)
    return ker / ker.sum()


def fz_P_matrix(N, ratio, kernel_length=3, sig=None, start=0):
    """Toeplitz(sqrt(diag(noyau 2D))) puis décimation — comme le .m."""
    sig = fz_sigma(ratio) if sig is None else sig
    ker = gaussian_kernel2d(kernel_length, sig)
    veck = np.sqrt(np.diag(ker))
    half_up = veck[kernel_length // 2:]            # centre → fin
    half_lo = veck[:kernel_length // 2]            # début (wrap en fin de ligne)
    col = np.concatenate([half_up, np.zeros(N - kernel_length), half_lo])
    P = toeplitz(col, col)
    P = P[start::int(round(ratio)), :]
    P /= P.sum(axis=1, keepdims=True) + 1e-12
    return P


def srf_nearest(bands_msi):
    """R (c × 21) : chaque bande MSI pointe la bande S3 la plus proche."""
    R = np.zeros((len(bands_msi), len(S3_BANDS)))
    for i, b in enumerate(bands_msi):
        R[i, np.argmin(np.abs(S3_BANDS - b))] = 1.0
    return R


def build_sources(streams, size=1500):
    """Sources (X, [P1, P2, P3], lam) sur la grille SRI 1500×1500×21."""
    I = np.eye(size)
    ops = {}
    for name, ratio in [('s2_20', 2), ('s2_60', 6), ('s3', 30)]:
        ops[name] = fz_P_matrix(size, ratio)
    srcs = [
        (streams['s2_10'], [I, I, srf_nearest(S2_BANDS[S2_RES == 10])], 1.0),
        (streams['s2_20'], [ops['s2_20'], ops['s2_20'],
                            srf_nearest(S2_BANDS[S2_RES == 20])], 1.0),
        (streams['s2_60'], [ops['s2_60'], ops['s2_60'],
                            srf_nearest(S2_BANDS[S2_RES == 60])], 1.0),
        (streams['s3'], [ops['s3'], ops['s3'], np.eye(len(S3_BANDS))], 1.0),
    ]
    return srcs


# ── Modèle torch multi-flux (généralisation de fuse_linear_torch) ─────────────

def fuse_linear_torch_multi(sources, ranks=(60, 60, 15), beta=1e-1, iters=4000,
                            lr=1e-2, log_every=50):
    """Tucker couplé sparse, N sources, Adam + prox (seuil découplé)."""
    tsrc = [(torch.tensor(X, dtype=torch.float32, device=device),
             [torch.tensor(P, dtype=torch.float32, device=device) for P in Ps],
             lam) for X, Ps, lam in sources]

    def unfold_t(T, mode):
        if mode == 0:
            return T.reshape(T.shape[0], -1)
        if mode == 1:
            return T.permute(1, 0, 2).reshape(T.shape[1], -1)
        return T.permute(2, 0, 1).reshape(T.shape[2], -1)

    def norm_cols(D):
        return D / (D.norm(dim=0, keepdim=True) + 1e-12)

    r1, r2, r3 = ranks
    M1, H = tsrc[0][0], tsrc[-1][0]
    with torch.no_grad():
        U1, _, _ = torch.linalg.svd(unfold_t(M1, 0), full_matrices=False)
        U2, _, _ = torch.linalg.svd(unfold_t(M1, 1), full_matrices=False)
        U3, _, _ = torch.linalg.svd(unfold_t(H, 2), full_matrices=False)
        D1 = norm_cols(U1[:, :r1])
        D2 = norm_cols(U2[:, :r2])
        D3h = norm_cols(U3[:, :r3])
        R1 = tsrc[0][1][2]
        G = torch.einsum('ijk,ia,jb,kc->abc', M1, D1, D2, R1 @ D3h)
        G = G / (G.norm() + 1e-12)

    G = G.clone().requires_grad_(True)
    D1 = D1.clone().requires_grad_(True)
    D2 = D2.clone().requires_grad_(True)
    D3 = D3h.clone().requires_grad_(True)
    params = [G, D1, D2, D3]
    opt = torch.optim.Adam(params, lr=lr)

    t0 = time.time()
    best_loss, best = float('inf'), None
    for it in range(1, iters + 1):
        opt.zero_grad()
        d1, d2, d3 = norm_cols(D1), norm_cols(D2), norm_cols(D3)
        loss = 0.0
        for X, (P1, P2, P3), lam in tsrc:
            # dégradation en espace facteur (jamais la SRI complète en mémoire)
            f1, f2, f3 = P1 @ d1, P2 @ d2, P3 @ d3
            pred = torch.einsum('abc,ia,jb,kc->ijk', G, f1, f2, f3)
            loss = loss + lam * ((pred - X) ** 2).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        with torch.no_grad():
            t = beta * lr
            G.copy_(torch.sign(G) * torch.clamp(G.abs() - t, min=0.0))
        lv = float(loss.item())
        if lv < best_loss:
            best_loss = lv
            best = [p.detach().clone() for p in params]
        if it % log_every == 0:
            sp = float((G.abs() < 1e-10).float().mean()) * 100
            print(f'    it {it:05d} | loss {lv:.4e} | sp(G) {sp:.1f}%', flush=True)

    Gb, D1b, D2b, D3b = best
    d1, d2, d3 = norm_cols(D1b), norm_cols(D2b), norm_cols(D3b)
    S = torch.einsum('abc,ia,jb,kc->ijk', Gb, d1, d2, d3).cpu().numpy()
    sparsity = float((Gb.abs() < 1e-10).float().mean()) * 100
    return np.clip(S, 0.0, 1.0), sparsity, time.time() - t0


# ── Métriques ─────────────────────────────────────────────────────────────────

def metrics(ref, pred, ratio=30):
    err = ref - pred
    psnr = 10 * np.log10(1.0 / max(np.mean(err ** 2), 1e-12))
    p = pred.reshape(-1, pred.shape[-1])
    t = ref.reshape(-1, ref.shape[-1])
    cos = np.sum(p * t, axis=1) / (np.linalg.norm(p, axis=1)
                                   * np.linalg.norm(t, axis=1) + 1e-9)
    sam = float(np.degrees(np.mean(np.arccos(np.clip(cos, -1, 1)))))
    means = ref.reshape(-1, ref.shape[-1]).mean(axis=0)
    rmse_b = np.sqrt(np.mean(err.reshape(-1, err.shape[-1]) ** 2, axis=0))
    ergas = 100.0 / ratio * np.sqrt(np.mean((rmse_b / (means + 1e-9)) ** 2))
    q2n = _calc_q2n(ref, pred)
    return dict(PSNR=float(psnr), SAM=sam, ERGAS=float(ergas), q2n=float(q2n))


if __name__ == '__main__':
    ranks = tuple(int(v) for v in (sys.argv[1].split(',') if len(sys.argv) > 1
                                   else ('60', '60', '15')))
    beta = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-1
    als_outer = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    print('Simulation des flux (protocole équipe : 1500², bilinéaire, sans bruit)...')
    streams, ref = simulate_streams()
    for k, v in streams.items():
        print(f'  {k}: {v.shape}')
    print(f'  ref : {ref.shape}')
    sources = build_sources(streams)

    results = {}
    print(f'\n=== Adam+prox multi-flux (rangs {ranks}, beta={beta}) ===', flush=True)
    S, sp, dt = fuse_linear_torch_multi(sources, ranks=ranks, beta=beta)
    m = metrics(ref, S)
    m.update(sparsity_G=sp, time_s=dt)
    results['adam_prox_multi'] = m
    print(' ', m, flush=True)

    print('\n=== ALS-Lipschitz v0 multi-flux ===', flush=True)
    try:
        from modeles_lineaires.v0_lineaire_baseline_tucker_als import run_gscott_tucker
        t0 = time.time()
        S2, G2, _, _ = run_gscott_tucker(sources, list(ranks), beta_factor=0.01,
                                         n_outer=als_outer, n_fista=60, verbose=True)
        m2 = metrics(ref, np.clip(S2, 0, 1))
        m2.update(sparsity_G=float(np.mean(np.abs(G2) < 1e-8)) * 100,
                  time_s=time.time() - t0)
        results['als_lip_multi'] = m2
        print(' ', m2, flush=True)
    except Exception as e:
        print('  ALS échoué :', e)

    import json
    with open('results_protocol_fz.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nok : results_protocol_fz.json')
    print('\nRappel tableau équipe (G-STEREO-1, Pavia) : '
          'PSNR 32.13 | SAM 8.27 | Q2n 0.59 | ERGAS 0.78')
