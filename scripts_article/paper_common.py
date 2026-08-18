"""
paper_common.py
─────────────────────────────────────────────────────────────────────────────
Module commun du pipeline de l'article NL-JTAE :
  - chargement PaviaU et simulation des 4 flux natifs (protocole de Wald),
  - opérateurs physiques PSF (flou gaussien + décimation) et SRF,
  - dégradations différentiables (PyTorch) pour l'apprentissage auto-supervisé,
  - métriques d'évaluation : PSNR, SAM, ERGAS, SSIM, UIQI + parcimonie de G,
  - perte SAM différentiable.

Régime d'information : les fonctions de simulation ne renvoient que les
observations et les opérateurs ; la vérité terrain S n'est JAMAIS utilisée
dans les pertes des modèles, uniquement dans evaluate_all().
─────────────────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import numpy as np
import scipy.io as sio
import torch

# ── Chargement des données ────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_paviau(crop_size=(240, 240)):
    """Charge PaviaU, normalise dans [0,1] (percentile 99.9) et recadre au centre."""
    path = os.path.join(DATA_DIR, "PaviaU.mat")
    img = sio.loadmat(path)["paviaU"].astype(np.float32)
    img = np.clip(img / np.percentile(img, 99.9), 0, 1)
    H, W, _ = img.shape
    h, w = crop_size
    sh, sw = (H - h) // 2, (W - w) // 2
    return img[sh:sh + h, sw:sw + w, :]


# ── Opérateurs physiques ──────────────────────────────────────────────────────

def gaussian_blur_matrix(dim, scale, sigma=1.0):
    """Matrice 1D de PSF gaussienne + décimation d'un facteur `scale`."""
    if scale == 1:
        return np.eye(dim)
    radius = int(4 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    B = np.zeros((dim, dim))
    for i in range(dim):
        for j, val in enumerate(kernel):
            col = i + x[j]
            if 0 <= col < dim:
                B[i, col] += val
    B /= B.sum(axis=1, keepdims=True)
    return B[::scale, :]


def spectral_response_matrix(C_bands, c_bands):
    """SRF simulée : agrégation uniforme par blocs de bandes contiguës."""
    R = np.zeros((c_bands, C_bands))
    step = C_bands / c_bands
    for i in range(c_bands):
        start = int(i * step)
        end = int((i + 1) * step) if i < c_bands - 1 else C_bands
        R[i, start:end] = 1.0 / (end - start)
    return R


def simulate_streams(S, sigma=1.0):
    """
    Protocole de Wald multi-échelle : génère les 4 flux natifs à partir de la
    scène latente S (H, W, C) :
      Y10 : 4 bandes @ 10 m  (H,   W)    — dégradation spectrale seule
      Y20 : 6 bandes @ 20 m  (H/2, W/2)
      Y60 : 3 bandes @ 60 m  (H/6, W/6)
      Y300: 21 bandes @ 300 m (H/30, W/30)
    Renvoie (obs, ops) : observations et opérateurs physiques uniquement.
    """
    H, W, C = S.shape
    ops = {}
    for name, scale in [("20", 2), ("60", 6), ("300", 30)]:
        ops[f"Ph_{name}"] = gaussian_blur_matrix(H, scale, sigma)
        ops[f"Pw_{name}"] = gaussian_blur_matrix(W, scale, sigma)
    ops["R_10"] = spectral_response_matrix(C, 4)
    ops["R_20"] = spectral_response_matrix(C, 6)
    ops["R_60"] = spectral_response_matrix(C, 3)
    ops["R_300"] = spectral_response_matrix(C, 21)

    def degrade(S, Ph, Pw, R):
        X = np.einsum('ij,jkl->ikl', Ph, S)
        X = np.einsum('ijk,lj->ilk', X, Pw)
        return np.einsum('ijk,lk->ijl', X, R)

    obs = {
        "Y10": np.einsum('ijk,lk->ijl', S, ops["R_10"]),
        "Y20": degrade(S, ops["Ph_20"], ops["Pw_20"], ops["R_20"]),
        "Y60": degrade(S, ops["Ph_60"], ops["Pw_60"], ops["R_60"]),
        "Y300": degrade(S, ops["Ph_300"], ops["Pw_300"], ops["R_300"]),
    }
    return obs, ops


# ── Dégradations différentiables (PyTorch, format B,H,W,C) ────────────────────

def torch_spatial_degrade(S_bhwc, Ph, Pw):
    """Applique P_h (mode 1) et P_w (mode 2) à un tenseur (B, H, W, C)."""
    X = S_bhwc.permute(0, 3, 1, 2)                       # (B, C, H, W)
    X = torch.einsum('ij,bcjl->bcil', Ph, X)             # (B, C, h, W)
    X = torch.einsum('bcil,kl->bcik', X, Pw)             # (B, C, h, w)
    return X.permute(0, 2, 3, 1)                         # (B, h, w, C)


def torch_degrade_all(S_bhwc, bufs):
    """Simule les 4 flux à partir de la prédiction (B, H, W, C)."""
    y10 = torch.matmul(S_bhwc, bufs["R_10"].T)
    y20 = torch.matmul(torch_spatial_degrade(S_bhwc, bufs["Ph_20"], bufs["Pw_20"]), bufs["R_20"].T)
    y60 = torch.matmul(torch_spatial_degrade(S_bhwc, bufs["Ph_60"], bufs["Pw_60"]), bufs["R_60"].T)
    y300 = torch.matmul(torch_spatial_degrade(S_bhwc, bufs["Ph_300"], bufs["Pw_300"]), bufs["R_300"].T)
    return y10, y20, y60, y300


def sam_loss(y_pred, y_true, eps=1e-8):
    """Perte d'angle spectral différentiable (radians moyens)."""
    p = y_pred.reshape(-1, y_pred.shape[-1])
    t = y_true.reshape(-1, y_true.shape[-1])
    cos = torch.sum(p * t, dim=1) / (torch.norm(p, dim=1) * torch.norm(t, dim=1) + eps)
    return torch.mean(torch.acos(torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)))


# ── Métriques d'évaluation (numpy, mêmes conventions que v1–v3) ───────────────

def calc_psnr(ref, fused):
    rmse = np.sqrt(np.mean((ref - fused) ** 2))
    return 100.0 if rmse == 0 else float(20 * np.log10(1.0 / rmse))


def calc_sam(ref, fused):
    r = ref.reshape(-1, ref.shape[-1])
    f = fused.reshape(-1, fused.shape[-1])
    dot = np.sum(r * f, axis=1)
    norms = np.linalg.norm(r, axis=1) * np.linalg.norm(f, axis=1)
    cos = np.clip(dot / (norms + 1e-8), -1.0, 1.0)
    return float(np.mean(np.arccos(cos)) * 180.0 / np.pi)


def calc_ergas(ref, fused, ratio=30):
    rmse_b = np.sqrt(np.mean((ref - fused) ** 2, axis=(0, 1)))
    mean_b = np.mean(ref, axis=(0, 1))
    val = np.sum((rmse_b / (mean_b + 1e-8)) ** 2)
    return float(100.0 / ratio * np.sqrt(val / ref.shape[-1]))


def calc_ssim(ref, fused):
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        return float("nan")
    vals = [ssim_fn(ref[..., b], fused[..., b], data_range=1.0)
            for b in range(ref.shape[-1])]
    return float(np.mean(vals))


def calc_uiqi(ref, fused):
    vals = []
    for b in range(ref.shape[-1]):
        x, y = ref[..., b], fused[..., b]
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = np.mean((x - mx) * (y - my))
        denom = (vx + vy) * (mx ** 2 + my ** 2)
        vals.append(1.0 if denom == 0 and vx == vy else 4.0 * cov * mx * my / denom if denom != 0 else 0.0)
    return float(np.mean(vals))


def calc_q2n(ref, fused, ws=32):
    """Calcul de l'index Q2n hypercomplexe en pure NumPy (généralisation de UIQI)."""
    ref = ref.astype(np.float64)
    fused = fused.astype(np.float64)
    H, W, N3 = ref.shape
    stride = ws

    stepx = int(np.ceil(H / stride))
    stepy = int(np.ceil(W / stride))
    if stepy <= 0:
        stepy = stepx = 1

    est1 = (stepx - 1) * stride + ws - H
    est2 = (stepy  - 1) * stride + ws - W

    if est1 != 0 or est2 != 0:
        ref_p = np.zeros((H + est1, W + est2, N3))
        fused_p  = np.zeros((H + est1, W + est2, N3))
        ref_p[:H, :W, :] = ref
        fused_p[:H, :W, :] = fused
        if est2 > 0:
            ref_p[:H, W:, :] = ref[:, W-est2:W, :][:, ::-1, :]
            fused_p[:H,  W:, :] = fused[:,  W-est2:W, :][:, ::-1, :]
        if est1 > 0:
            ref_p[H:, :, :] = ref_p[H-est1:H, :, :][::-1, :, :]
            fused_p[H:,  :, :] = fused_p[ H-est1:H, :, :][::-1, :, :]
        ref, fused = ref_p, fused_p
        H, W = ref.shape[:2]

    n_pow2 = 1
    while n_pow2 < N3:
        n_pow2 *= 2
    if n_pow2 > N3:
        pad = np.zeros((H, W, n_pow2 - N3))
        ref = np.concatenate([ref, pad], axis=2)
        fused  = np.concatenate([fused,  pad], axis=2)
    N3 = n_pow2

    def _norm_blocco(x):
        a = x.mean()
        c = x.std(ddof=1)
        if c == 0:
            c = np.finfo(float).eps
        return (x - a) / c + 1.0, a, c

    def _onion_mult2D(o1, o2):
        n3 = o1.shape[2]
        if n3 == 1:
            return o1 * o2
        L = n3 // 2
        a = o1[:,:,:L];  b = np.concatenate([o1[:,:,L:L+1], -o1[:,:,L+1:]], axis=2)
        c = o2[:,:,:L];  d = np.concatenate([o2[:,:,L:L+1], -o2[:,:,L+1:]], axis=2)
        if n3 == 2:
            return np.concatenate([a*c - d*b, a*d + c*b], axis=2)
        ris1 = _onion_mult2D(a, c)
        ris2 = _onion_mult2D(d, np.concatenate([b[:,:,:1], -b[:,:,1:]], axis=2))
        ris3 = _onion_mult2D(np.concatenate([a[:,:,:1], -a[:,:,1:]], axis=2), d)
        ris4 = _onion_mult2D(c, b)
        return np.concatenate([ris1 - ris2, ris3 + ris4], axis=2)

    def _onion_mult(o1, o2):
        n = len(o1)
        if n == 1:
            return o1 * o2
        L = n // 2
        a = o1[:L];  b = np.concatenate([[o1[L]], -o1[L+1:]])
        c = o2[:L];  d = np.concatenate([[o2[L]], -o2[L+1:]])
        if n == 2:
            return np.array([a[0]*c[0] - d[0]*b[0], a[0]*d[0] + c[0]*b[0]])
        ris1 = _onion_mult(a, c)
        ris2 = _onion_mult(d, np.concatenate([[b[0]], -b[1:]]))
        ris3 = _onion_mult(np.concatenate([[a[0]], -a[1:]]), d)
        ris4 = _onion_mult(c, b)
        return np.concatenate([ris1 - ris2, ris3 + ris4])

    def _onions_quality(dat1, dat2, ws_val):
        dat1 = dat1.astype(np.float64)
        dat2 = np.concatenate([dat2[:,:,:1], -dat2[:,:,1:]], axis=2).astype(np.float64)
        n3 = dat1.shape[2]
        M  = ws_val * ws_val
        m1 = np.zeros(n3);  m2 = np.zeros(n3)
        mod_q1m = 0.0;      mod_q2m = 0.0
        mod_q1  = np.zeros((ws_val, ws_val));  mod_q2 = np.zeros((ws_val, ws_val))
        for i in range(n3):
            a1, s, t = _norm_blocco(dat1[:,:,i])
            dat1[:,:,i] = a1
            if s == 0:
                dat2[:,:,i] = dat2[:,:,i] - s + 1 if i == 0 else -(-dat2[:,:,i] - s + 1)
            else:
                dat2[:,:,i] = (dat2[:,:,i] - s) / t + 1 if i == 0 else -((-dat2[:,:,i] - s) / t + 1)
            m1[i] = dat1[:,:,i].mean();  m2[i] = dat2[:,:,i].mean()
            mod_q1m += m1[i]**2;         mod_q2m += m2[i]**2
            mod_q1  += dat1[:,:,i]**2;   mod_q2  += dat2[:,:,i]**2
        mod_q1m = np.sqrt(mod_q1m);  mod_q2m = np.sqrt(mod_q2m)
        mod_q1  = np.sqrt(mod_q1);   mod_q2  = np.sqrt(mod_q2)
        termine2 = mod_q1m * mod_q2m
        termine4 = mod_q1m**2 + mod_q2m**2
        termine3 = (M/(M-1)) * (np.mean(mod_q1**2) + np.mean(mod_q2**2)) - (M/(M-1)) * termine4
        mean_bias = 2.0 * termine2 / termine4 if termine4 > 0 else 0.0
        if termine3 == 0:
            q = np.zeros(n3);  q[n3-1] = mean_bias
        else:
            cbm = 2.0 / termine3
            qu  = _onion_mult2D(dat1, dat2)
            qm  = _onion_mult(m1, m2)
            qv  = np.array([(M/(M-1)) * np.mean(qu[:,:,i]) for i in range(n3)])
            q   = (qv - (M/(M-1)) * qm) * mean_bias * cbm
        return q

    valori = np.zeros((stepx, stepy, N3))
    for j in range(stepx):
        for i in range(stepy):
            valori[j, i, :] = _onions_quality(
                ref[j*stride:j*stride+ws, i*stride:i*stride+ws, :],
                fused[j*stride:j*stride+ws, i*stride:i*stride+ws, :],
                ws)

    return float(np.mean(np.sqrt(np.sum(valori**2, axis=2))))


def core_sparsity(G, tol=1e-4):
    """Pourcentage de coefficients de G quasi nuls (|g| < tol)."""
    if isinstance(G, torch.Tensor):
        return float((G.abs() < tol).float().mean().item()) * 100
    return float(np.mean(np.abs(G) < tol)) * 100


def evaluate_all(S_ref, S_hat, G=None, ratio=30):
    """Panel complet de métriques contre la vérité terrain (évaluation seule)."""
    S_hat = np.clip(S_hat, 0, 1)
    out = {
        "PSNR": calc_psnr(S_ref, S_hat),
        "SAM": calc_sam(S_ref, S_hat),
        "ERGAS": calc_ergas(S_ref, S_hat, ratio),
        "SSIM": calc_ssim(S_ref, S_hat),
        "UIQI": calc_uiqi(S_ref, S_hat),
        "Q2n": calc_q2n(S_ref, S_hat),
    }
    if G is not None:
        out["Sparsity_G"] = core_sparsity(G)
    return out


def format_metrics(name, m):
    line = (f"{name:<28s} | PSNR {m['PSNR']:6.2f} dB | SAM {m['SAM']:5.2f}° | "
            f"ERGAS {m['ERGAS']:7.4f} | SSIM {m['SSIM']:6.4f} | Q2n {m['Q2n']:6.4f}")
    if "Sparsity_G" in m:
        line += f" | Sp(G) {m['Sparsity_G']:5.1f}%"
    return line
