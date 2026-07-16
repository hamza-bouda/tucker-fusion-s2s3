"""
paper_v1_ctae.py
─────────────────────────────────────────────────────────────────────────────
Auto-encodeur de Tucker couplé (CTAE) : encodeurs CNN non-linéaires par
capteur estimant un cœur partagé, décodage MULTILINÉAIRE par dictionnaires
appris A (H×R1), B (W×R2), C (C_bands×R3).

Hybride non-linéaire/linéaire : l'analyse (encodeurs) est non-linéaire, la
synthèse reste un produit de Tucker — c'est la limite que le NL-JTAE lève.

Régime auto-supervisé : pertes sur les 4 flux natifs via PSF/SRF + couplage
des deux cœurs + L1(G). Initialisation SVD depuis les observations.
─────────────────────────────────────────────────────────────────────────────
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paper_common import (load_paviau, simulate_streams, torch_degrade_all,
                          evaluate_all, format_metrics)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        b = out_ch // 4
        def branch(*layers):
            return nn.Sequential(*layers)
        self.b1 = branch(nn.Conv2d(in_ch, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.b2 = branch(nn.Conv2d(in_ch, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.b3 = branch(nn.Conv2d(in_ch, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b, b, 5, padding=2), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.b4 = branch(nn.MaxPool2d(3, stride=1, padding=1),
                         nn.Conv2d(in_ch, b, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class CoreEncoder(nn.Module):
    """Encodeur CNN → cœur (B, R1, R2, R3) par pooling adaptatif."""
    def __init__(self, in_ch, R3, grid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.inception = InceptionBlock(64, 64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, R3, 3, padding=1), nn.BatchNorm2d(R3), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(grid)

    def forward(self, x):
        h = self.net(x)
        h = h + self.inception(h)
        return self.pool(self.head(h)).permute(0, 2, 3, 1)  # (B, R1, R2, R3)


class CoupledTuckerAE(nn.Module):
    def __init__(self, H, W, C_bands, ranks, ops, init_factors):
        super().__init__()
        R1, R2, R3 = ranks
        for k, v in ops.items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32))
        self.enc_S2 = CoreEncoder(13, R3, (R1, R2))
        self.enc_S3 = CoreEncoder(21, R3, (R1, R2))
        A0, B0, C0 = init_factors
        self.A = nn.Parameter(torch.tensor(A0, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(B0, dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(C0, dtype=torch.float32))

    def bufs(self):
        return {k: getattr(self, k) for k in
                ["Ph_20", "Pw_20", "Ph_60", "Pw_60", "Ph_300", "Pw_300",
                 "R_10", "R_20", "R_60", "R_300"]}

    def forward(self, x_S2_up, x_S3_up):
        G_S2 = self.enc_S2(x_S2_up)
        G_S3 = self.enc_S3(x_S3_up)
        # Synthèse strictement multilinéaire (produit de Tucker)
        S_pred = torch.einsum('brst,ir,js,kt->bijk', G_S2, self.A, self.B, self.C)
        return S_pred, G_S2, G_S3


def svd_init_factors(obs, ops, ranks):
    """Initialisation des dictionnaires depuis les observations uniquement."""
    R1, R2, R3 = ranks
    Y10, Y300 = obs["Y10"], obs["Y300"]
    A0 = np.linalg.svd(Y10.reshape(Y10.shape[0], -1), full_matrices=False)[0][:, :R1]
    B0 = np.linalg.svd(Y10.transpose(1, 0, 2).reshape(Y10.shape[1], -1),
                       full_matrices=False)[0][:, :R2]
    U3 = np.linalg.svd(Y300.transpose(2, 0, 1).reshape(21, -1), full_matrices=False)[0][:, :R3]
    C0 = np.linalg.pinv(ops["R_300"]) @ U3
    return A0, B0, C0


def run_ctae(S, obs, ops, ranks=(24, 24, 12), epochs=1500, lr=1e-3,
             lam_couple=0.1, lam_sparse=1e-5, patience=250, log_every=100, verbose=True):
    H, W, C = S.shape

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32)

    # Entrées : flux interpolés bilinéairement à la grille 10 m (design CTAE)
    y10 = to_t(obs["Y10"]).permute(2, 0, 1).unsqueeze(0)
    y20 = F.interpolate(to_t(obs["Y20"]).permute(2, 0, 1).unsqueeze(0), size=(H, W), mode='bilinear')
    y60 = F.interpolate(to_t(obs["Y60"]).permute(2, 0, 1).unsqueeze(0), size=(H, W), mode='bilinear')
    x_S2 = torch.cat([y10, y20, y60], dim=1).to(device)
    x_S3 = F.interpolate(to_t(obs["Y300"]).permute(2, 0, 1).unsqueeze(0),
                         size=(H, W), mode='bilinear').to(device)

    targets = {k: to_t(obs[k]).unsqueeze(0).to(device) for k in obs}

    model = CoupledTuckerAE(H, W, C, ranks, ops, svd_init_factors(obs, ops, ranks)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=30)

    best_loss, best_S, best_G, stall = float('inf'), None, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        S_pred, G_S2, G_S3 = model(x_S2, x_S3)
        p10, p20, p60, p300 = torch_degrade_all(S_pred, model.bufs())
        loss = (F.mse_loss(p10, targets["Y10"]) + F.mse_loss(p20, targets["Y20"])
                + F.mse_loss(p60, targets["Y60"]) + 2.0 * F.mse_loss(p300, targets["Y300"])
                + lam_couple * F.mse_loss(G_S2, G_S3)
                + lam_sparse * G_S2.abs().mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss.item())

        if loss.item() < best_loss - 1e-8:
            best_loss, stall = loss.item(), 0
            best_S = S_pred.detach().squeeze(0).cpu().numpy()
            best_G = G_S2.detach().cpu()
        else:
            stall += 1
            if stall >= patience:
                if verbose:
                    print(f"  [early stopping] époque {ep}")
                break
        if verbose and (ep % log_every == 0 or ep == 1):
            print(f"  ep {ep:04d} | loss {loss.item():.3e}")
    elapsed = time.time() - t0
    return np.clip(best_S, 0, 1), best_G, elapsed


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="CTAE couplé auto-supervisé (v1)")
    p.add_argument('--ranks', type=int, nargs=3, default=[24, 24, 12])
    p.add_argument('--epochs', type=int, default=1500)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()

    S = load_paviau()
    obs, ops = simulate_streams(S)
    S_hat, G, elapsed = run_ctae(S, obs, ops, ranks=tuple(args.ranks),
                                 epochs=args.epochs, lr=args.lr)
    m = evaluate_all(S, S_hat, G)
    m["time_s"] = elapsed
    print("\n" + format_metrics("v1 CTAE (couplé)", m))
    with open("results/paper_v1_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
