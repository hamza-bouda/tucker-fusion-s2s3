"""
linear_tucker_torch.py
─────────────────────────────────────────────────────────────────────────────
Modèle LINÉAIRE G-SCOTT-Tucker (Tucker couplé à cœur sparse), résolu par
descente de gradient PyTorch au lieu de l'ALS+FISTA de la version historique.

Le modèle est strictement identique à la v0 :
    S = G ×1 D1 ×2 D2 ×3 D3
    L = Σk λk ‖Mk − dégradé_k(S)‖² + β ‖G‖₁
Seul l'algorithme d'optimisation change. Quatre variantes :

    adam_prox : Adam sur la partie lisse + seuillage doux proximal sur G
                après chaque pas (avec seuil découpé du scheduler,
                zéros EXACTS dans le cœur).
    adam_l1   : Adam avec la pénalité L1 directement dans la perte
                (sous-gradient — ne produit jamais de zéros exacts).
    sgd_prox  : SGD + momentum sur la partie lisse + même prox sur G.
    lbfgs_prox: LBFGS sur la partie lisse + même prox sur G.

Init SVD depuis les observations (identique à la v0), normalisation des
colonnes des dictionnaires à chaque itération (l'échelle vit dans G).
Aucune vérité terrain n'est utilisée.
─────────────────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import time
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

METHODS = ('adam_prox', 'adam_l1', 'sgd_prox', 'lbfgs_prox')


def build_psf_matrix(N, r, psf_kernel):
    """Matrice de dégradation spatiale 1D (N//r × N) depuis le noyau PSF 2D."""
    mid = psf_kernel.shape[0] // 2
    psf1d = psf_kernel[mid, :]
    psf1d = psf1d / (psf1d.sum() + 1e-12)
    half = len(psf1d) // 2
    B = np.zeros((N // r, N))
    for i in range(N // r):
        center = i * r
        for j, w in enumerate(psf1d):
            col = center + j - half
            if 0 <= col < N:
                B[i, col] += w
    B /= B.sum(axis=1, keepdims=True) + 1e-12
    return B


def left_svd(M, k):
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    return U[:, :min(k, U.shape[1])]


def unfold(T, mode):
    if mode == 0:
        return T.reshape(T.shape[0], -1)
    if mode == 1:
        return T.permute(1, 0, 2).reshape(T.shape[1], -1)
    return T.permute(2, 0, 1).reshape(T.shape[2], -1)


def norm_cols(D):
    return D / (D.norm(dim=0, keepdim=True) + 1e-12)


def reconstruct(G, D1, D2, D3):
    return torch.einsum('abc,ia,jb,kc->ijk', G, D1, D2, D3)


def fuse_linear_torch(hr_msi, lr_hsi, srf, psf_kernel, method='adam_prox',
                      ranks=(15, 15, 8), beta=1e-6, lam_H=1.0, lam_M=1.0,
                      iters=3000, lr=1e-2, log_every=10,
                      scheduler_type='constant', patience=10000, tol_es=1e-7,
                      prox_step=None):
    """Fusion linéaire par gradient. Retourne (S, stats) avec l'historique."""
    assert method in METHODS, method
    Hs, Ws, c = hr_msi.shape
    hs, ws, C = lr_hsi.shape
    r_sp = Hs // hs

    # Normalisation (conventions de la version ALS historique)
    s_M = np.percentile(hr_msi, 99) + 1e-9
    s_H = np.percentile(lr_hsi, 99) + 1e-9
    M1 = torch.tensor(hr_msi / s_M, dtype=torch.float32, device=device)
    Hn = torch.tensor(lr_hsi / s_H, dtype=torch.float32, device=device)
    scale = s_H / s_M   # cohérence d'échelle : R·(S/s_H) vs M1/s_M

    Bh = torch.tensor(build_psf_matrix(Hs, r_sp, psf_kernel), dtype=torch.float32, device=device)
    Bw = torch.tensor(build_psf_matrix(Ws, r_sp, psf_kernel), dtype=torch.float32, device=device)
    R = torch.tensor(srf, dtype=torch.float32, device=device)          # (c, C)

    # ── Init SVD (identique v0 : dictionnaires depuis les observations) ──────
    r1, r2, r3 = ranks
    with torch.no_grad():
        D1 = norm_cols(left_svd(unfold(M1, 0), r1))
        D2 = norm_cols(left_svd(unfold(M1, 1), r2))
        D3 = norm_cols(left_svd(unfold(Hn, 2), r3))
        G = torch.einsum('ijk,ia,jb,kc->abc', M1, D1, D2, R @ D3)
        G = G / (G.norm() + 1e-12)

    G = G.clone().requires_grad_(True)
    D1 = D1.clone().requires_grad_(True)
    D2 = D2.clone().requires_grad_(True)
    D3 = D3.clone().requires_grad_(True)
    params = [G, D1, D2, D3]
    for p in params:
        p.register_hook(lambda grad: grad.contiguous())

    if method == 'sgd_prox':
        opt = torch.optim.SGD(params, lr=lr * 0.5, momentum=0.9)
    elif method.startswith('lbfgs'):
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=20, tolerance_grad=1e-6, history_size=10)
    else:
        opt = torch.optim.Adam(params, lr=lr)

    scheduler = None
    if scheduler_type == 'cosine' and not method.startswith('lbfgs'):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-5)

    def smooth_loss():
        d1, d2, d3 = norm_cols(D1), norm_cols(D2), norm_cols(D3)
        S = reconstruct(G, d1, d2, d3)
        # Source HSI : PSF + décimation, spectre complet
        pred_H = torch.einsum('ij,jwc->iwc', Bh, S)
        pred_H = torch.einsum('iwc,kw->ikc', pred_H, Bw)
        # Source MSI : projection SRF, pleine résolution
        pred_M = torch.einsum('hwc,mc->hwm', S, R) * scale
        return (lam_H * ((pred_H - Hn) ** 2).sum()
                + lam_M * ((pred_M - M1) ** 2).sum()), S

    t0 = time.time()
    hist = {'iter': [], 'loss': [], 'sparsity': []}
    best_loss, best_S = float('inf'), None
    patience_counter = 0

    for it in range(1, iters + 1):
        if method.startswith('lbfgs'):
            def closure():
                opt.zero_grad()
                loss_val, _ = smooth_loss()
                if method == 'lbfgs_l1':
                    loss_val = loss_val + beta * G.abs().sum()
                loss_val.backward()
                # LBFGS fait p.grad.view(-1) en interne : exige des gradients
                # contigus, ce que les einsum ne garantissent pas
                for p in params:
                    if p.grad is not None and not p.grad.is_contiguous():
                        p.grad = p.grad.contiguous()
                return loss_val
            opt.step(closure)
            with torch.no_grad():
                loss, S = smooth_loss()
                if method == 'lbfgs_l1':
                    loss = loss + beta * G.abs().sum()
        else:
            opt.zero_grad()
            loss, S = smooth_loss()
            if method == 'adam_l1':
                loss = loss + beta * G.abs().sum()
            loss.backward()
            
            grad_norm = 0.0
            for p in params:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            
            if grad_norm < 1e-6:
                break

        if method in ('adam_prox', 'sgd_prox', 'lbfgs_prox'):
            with torch.no_grad():
                # Decoupled threshold: use the initial lr (or a specified prox_step) instead of scheduled lr
                step_size = prox_step if prox_step is not None else lr
                t = beta * step_size
                G.copy_(torch.sign(G) * torch.clamp(G.abs() - t, min=0.0))

        if scheduler is not None:
            scheduler.step()

        lv = float(loss.item())
        if lv < best_loss - tol_es:
            best_loss = lv
            best_S = S.detach()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if it % log_every == 0 or it == 1:
            sp = float((G.abs() < 1e-10).float().mean()) * 100
            hist['iter'].append(it)
            hist['loss'].append(lv)
            hist['sparsity'].append(sp)

    S_out = np.clip(best_S.cpu().numpy() * s_H, 0.0, 1.0)
    sparsity = float((G.detach().abs() < 1e-10).float().mean()) * 100
    stats = {'sparsity_G': sparsity, 'fit_time_s': time.time() - t0,
             'final_loss': best_loss, 'history': hist}
    return S_out, stats
