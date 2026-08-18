"""
nljtae_hyperbench.py
─────────────────────────────────────────────────────────────────────────────
Adaptateur HyperBench du NL-JTAE (étude de faisabilité, PaviaU).

HyperBench fournit 2 entrées par cas : hr_msi (H, W, c) et lr_hsi (h, w, C),
avec la SRF et le noyau PSF utilisés pour la dégradation. Cette variante
bi-flux généralise l'architecture multi-input : le ratio r = H/h est
factorisé en une cascade de strides (4 → 2×2, 8 → 2×2×2, ...), le MSI est
ingéré nativement au niveau 0, le HSI au niveau du cœur. Le cœur conjoint G
vit sur la grille basse résolution et sa parcimonie est structurelle
(seuillage doux borné, hors weight decay, avec warm-up).

Entraînement 100 % auto-supervisé par cas : fidélité SRF au MSI + fidélité
PSF au HSI + SAM sur le HSI simulé + L1 sur le pré-cœur. Aucune référence.
─────────────────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbench import BaseAdapter, ReconstructionInputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Utilitaires ───────────────────────────────────────────────────────────────

def factorize_ratio(r):
    """Décompose le ratio en strides premiers croissants (4→[2,2], 30→[2,3,5])."""
    strides, d = [], 2
    while r > 1:
        while r % d == 0:
            strides.append(d)
            r //= d
        d += 1
    return strides


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


def sam_loss(y_pred, y_true, eps=1e-8):
    p = y_pred.reshape(-1, y_pred.shape[-1])
    t = y_true.reshape(-1, y_true.shape[-1])
    cos = torch.sum(p * t, dim=1) / (torch.norm(p, dim=1) * torch.norm(t, dim=1) + eps)
    return torch.mean(torch.acos(torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)))


def conv_block(in_ch, out_ch, k=3):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, padding=k // 2),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class GatedFuse(nn.Module):
    def __init__(self, ch_a, ch_b, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(ch_a + ch_b, 2 * out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * out_ch)

    def forward(self, a, b):
        return F.glu(self.bn(self.conv(torch.cat([a, b], dim=1))), dim=1)


class SoftShrink(nn.Module):
    """Seuillage doux borné (tau = tau_max·sigmoïde(theta), hors weight decay)."""
    def __init__(self, ch, init=0.005, tau_max=0.05):
        super().__init__()
        self.tau_max = tau_max
        p = init / tau_max
        self.theta = nn.Parameter(torch.full((1, ch, 1, 1), float(np.log(p / (1 - p)))))
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return x
        tau = self.tau_max * torch.sigmoid(self.theta)
        return torch.sign(x) * F.relu(x.abs() - tau)


# ── Modèle bi-flux ────────────────────────────────────────────────────────────

class MSNLJTAE2(nn.Module):
    """NL-JTAE bi-flux : pyramide alignée sur la factorisation du ratio."""
    def __init__(self, c_msi, C_hsi, strides, r3=64):
        super().__init__()
        self.strides = strides
        n = len(strides)
        chs = [48] + [min(64 + 32 * i, 160) for i in range(n)]

        self.stem_msi = nn.Sequential(conv_block(c_msi, 48), conv_block(48, 48))
        self.stem_hsi = nn.Sequential(
            nn.Conv2d(C_hsi, chs[-1], 1), nn.BatchNorm2d(chs[-1]), nn.ReLU(inplace=True),
            conv_block(chs[-1], chs[-1]))

        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(chs[i], chs[i + 1], strides[i] + 2,
                                    stride=strides[i], padding=1),
                          nn.BatchNorm2d(chs[i + 1]), nn.ReLU(inplace=True))
            for i in range(n)])
        self.fuse = GatedFuse(chs[-1], chs[-1], chs[-1])

        self.core_head = nn.Sequential(
            nn.Conv2d(chs[-1], chs[-1], 1), nn.BatchNorm2d(chs[-1]), nn.ReLU(inplace=True),
            nn.Conv2d(chs[-1], r3, 1))
        self.shrink = SoftShrink(r3)

        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        in_ch = r3
        for i in reversed(range(n)):
            self.ups.append(nn.ConvTranspose2d(in_ch, chs[i], strides[i], stride=strides[i]))
            self.decs.append(conv_block(chs[i] * 2, chs[i]))
            in_ch = chs[i]
        self.head = nn.Sequential(nn.Conv2d(chs[0], C_hsi, 3, padding=1), nn.Sigmoid())

    def forward(self, msi, hsi):
        feats = [self.stem_msi(msi)]
        for down in self.downs:
            feats.append(down(feats[-1]))
        deep = self.fuse(feats[-1], self.stem_hsi(hsi))

        pre_core = self.core_head(deep)
        G = self.shrink(pre_core)

        x = G
        skips = feats[:-1][::-1]  # niveaux dérivés du MSI, du profond au fin
        for up, dec, skip in zip(self.ups, self.decs, skips):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x), G, pre_core


# ── Entraînement auto-supervisé par cas ───────────────────────────────────────

def fuse_nljtae(hr_msi, lr_hsi, srf, psf_kernel, r3=64, epochs=3000, lr=2e-3,
                lam_sam=0.1, lam_sparse=1e-6, patience=200, verbose=False):
    Hs, Ws, c = hr_msi.shape
    hs, ws, C = lr_hsi.shape
    r = Hs // hs
    hr_msi = hr_msi[:hs * r, :ws * r, :]           # aligner exactement les grilles
    Hs, Ws = hr_msi.shape[:2]
    strides = factorize_ratio(r)

    s_M = np.percentile(hr_msi, 99) + 1e-9
    s_H = np.percentile(lr_hsi, 99) + 1e-9
    msi = torch.tensor(hr_msi / s_M, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    hsi = torch.tensor(lr_hsi / s_H, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    Bh = torch.tensor(build_psf_matrix(Hs, r, psf_kernel), dtype=torch.float32).to(device)
    Bw = torch.tensor(build_psf_matrix(Ws, r, psf_kernel), dtype=torch.float32).to(device)
    R = torch.tensor(srf, dtype=torch.float32).to(device)      # (c, C)
    # Cohérence d'échelle entre les deux normalisations : R·(x/s_H) vs (y/s_M)
    scale = s_H / s_M

    model = MSNLJTAE2(c, C, strides, r3=r3).to(device)
    theta_id = id(model.shrink.theta)
    decay = [p for p in model.parameters() if p.ndim > 1 and id(p) != theta_id]
    no_decay = [p for p in model.parameters() if p.ndim <= 1] + [model.shrink.theta]
    opt = torch.optim.Adam([{"params": decay, "weight_decay": 1e-4},
                            {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=30)

    hsi_bhwc = hsi.permute(0, 2, 3, 1)
    warmup = 150
    best_loss, best_S, best_G, stall = float('inf'), None, None, 0
    for ep in range(1, epochs + 1):
        model.shrink.enabled = ep > warmup
        if ep == warmup + 1:
            best_loss, stall = float('inf'), 0
        model.train()
        opt.zero_grad()
        S_pred, G, pre_core = model(msi, hsi)           # (1, C, H, W) normalisé /s_H
        S_bhwc = S_pred.permute(0, 2, 3, 1)

        # Fidélité MSI : projection SRF (avec recalage d'échelle)
        p_msi = torch.matmul(S_bhwc, R.T) * scale        # (1, H, W, c)
        loss_msi = F.mse_loss(p_msi, msi.permute(0, 2, 3, 1))
        # Fidélité HSI : PSF + décimation
        p_hsi = torch.einsum('ij,bcjl->bcil', Bh, S_pred)
        p_hsi = torch.einsum('bcil,kl->bcik', p_hsi, Bw).permute(0, 2, 3, 1)
        loss_hsi = F.mse_loss(p_hsi, hsi_bhwc)

        loss = (loss_msi + 2.0 * loss_hsi + lam_sam * sam_loss(p_hsi, hsi_bhwc)
                + lam_sparse * pre_core.abs().mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss.item())

        if loss.item() < best_loss - 1e-8:
            best_loss, stall = loss.item(), 0
            best_S = S_pred.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
            best_G = G.detach()
        else:
            stall += 1
            if stall >= patience:
                break
        if verbose and ep % 200 == 0:
            sp = float((G.abs() < 1e-4).float().mean()) * 100
            print(f"    ep {ep:04d} | loss {loss.item():.3e} | sp(G) {sp:.1f}%")

    sparsity = float((best_G.abs() < 1e-4).float().mean()) * 100
    return np.clip(best_S * s_H, 0.0, 1.0), sparsity


# ── Adaptateur HyperBench ─────────────────────────────────────────────────────

class NLJTAEAdapter(BaseAdapter):
    """NL-JTAE bi-flux auto-supervisé, cœur Tucker sparse structurel."""
    def __init__(self, r3=64, epochs=3000, lr=2e-3, lam_sam=0.1, lam_sparse=1e-6):
        super().__init__(name='NL-JTAE (proposé)', shape_policy='crop')
        self.r3, self.epochs, self.lr = r3, epochs, lr
        self.lam_sam, self.lam_sparse = lam_sam, lam_sparse

    def predict(self, inputs: ReconstructionInputs):
        t0 = time.time()
        H0, W0 = np.array(inputs.hr_msi).shape[:2]
        S, sparsity = fuse_nljtae(
            hr_msi=np.array(inputs.hr_msi, dtype=np.float64),
            lr_hsi=np.array(inputs.lr_hsi, dtype=np.float64),
            srf=np.array(inputs.srf, dtype=np.float64),
            psf_kernel=np.array(inputs.psf, dtype=np.float64),
            r3=self.r3, epochs=self.epochs, lr=self.lr,
            lam_sam=self.lam_sam, lam_sparse=self.lam_sparse)
        # Le réseau exige des dimensions divisibles par le ratio : la fusion
        # travaille sur une grille recadrée, on repad aux dimensions attendues.
        if S.shape[:2] != (H0, W0):
            S = np.pad(S, ((0, H0 - S.shape[0]), (0, W0 - S.shape[1]), (0, 0)),
                       mode='edge')
        return S, {'sparsity_G': sparsity, 'fit_time_s': time.time() - t0}
