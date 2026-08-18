"""
paper_v4_msnljtae.py
─────────────────────────────────────────────────────────────────────────────
NL-JTAE multi-input natif (proposé) : auto-encodeur de Tucker conjoint 100 %
non-linéaire ingérant les 4 flux à leur résolution native, sans aucune
interpolation préalable des données.

Principe : le rapport d'échelle 30 = 2×3×5 fait coïncider la pyramide
d'encodage avec les grilles capteurs (10→20→60→300 m). Chaque flux est
injecté à son niveau natif par fusion « gated » (GLU). Le cœur conjoint G
vit sur la grille S3 (H/30 × W/30 × R3) et sa parcimonie est structurelle :
seuillage doux à seuils appris (opérateur proximal de la norme L1).

Routage de l'information : les skips ne portent que des caractéristiques
dérivées de S2 ; la richesse spectrale des 21 bandes ne peut transiter que
par G — le cœur est contraint par construction d'être le code conjoint.

Régime auto-supervisé : pertes sur les 4 flux natifs (PSF/SRF) + SAM sur le
flux S3 simulé (fidélité spectrale angulaire aux OBSERVATIONS, jamais à la
vérité terrain) + L1 sur G. Un mode supervisé « oracle » est disponible via
--supervised, uniquement pour borne supérieure explicitement étiquetée.
─────────────────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts_article.paper_common import (load_paviau, simulate_streams, torch_degrade_all,
                          sam_loss, evaluate_all, format_metrics, core_sparsity)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Briques ───────────────────────────────────────────────────────────────────

def conv_block(in_ch, out_ch, k=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k // 2),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class Down(nn.Module):
    """Descente d'échelle apprise ×s : conv striée k=s+2, p=1 (anti-aliasing)."""
    def __init__(self, in_ch, out_ch, s):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=s + 2, stride=s, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class GatedFuse(nn.Module):
    """Fusion multiplicative type GLU : z = φ(conv[a;b]) ⊙ σ(conv[a;b])."""
    def __init__(self, ch_a, ch_b, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(ch_a + ch_b, 2 * out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * out_ch)

    def forward(self, a, b):
        return F.glu(self.bn(self.conv(torch.cat([a, b], dim=1))), dim=1)


class SoftShrink(nn.Module):
    """Seuillage doux à seuils appris par canal (prox L1) → zéros exacts.

    Trois garde-fous contre la mort du cœur (zone morte à gradient nul) :
      1. seuil borné : tau = tau_max · sigmoïde(theta), tau_max petit ;
      2. theta doit être EXCLU du weight decay (sinon la décroissance le
         tire vers tau = tau_max/2 et égorge le cœur) ;
      3. warm-up : `enabled=False` au début, le cœur apprend avant élagage.
    La pénalité L1 de la perte s'applique au PRÉ-cœur (gradient vivant).
    """
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


# ── Modèle ────────────────────────────────────────────────────────────────────

class MSNLJTAE(nn.Module):
    """
    Encodeur : pyramide 10→20→60→300 m (strides 2, 3, 5), injection native.
    Cœur     : G ∈ (B, R3, H/30, W/30), sparse par seuillage doux appris.
    Décodeur : miroir ConvTranspose ×5, ×3, ×2 avec skips des niveaux fusionnés.
    """
    def __init__(self, C_bands, ops, r3=64, skip_dropout=0.0):
        super().__init__()
        self.r3 = r3
        self.skip_dropout = skip_dropout
        for k, v in ops.items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32))

        # Stems natifs (un par flux, à sa résolution d'origine)
        self.stem10 = nn.Sequential(conv_block(4, 48), conv_block(48, 48))
        self.stem20 = conv_block(6, 48)
        self.stem60 = conv_block(3, 64)
        self.stem300 = nn.Sequential(
            nn.Conv2d(21, 96, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            conv_block(96, 96))

        # Pyramide descendante avec fusion gated par niveau
        self.down0 = Down(48, 64, s=2)     # 10 m → 20 m
        self.fuse1 = GatedFuse(64, 48, 64)
        self.down1 = Down(64, 96, s=3)     # 20 m → 60 m
        self.fuse2 = GatedFuse(96, 64, 96)
        self.down2 = Down(96, 128, s=5)    # 60 m → 300 m
        self.fuse3 = GatedFuse(128, 96, 128)

        # Tête cœur : projection spectrale non-linéaire + parcimonie structurelle
        self.core_head = nn.Sequential(
            nn.Conv2d(128, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, r3, 1))
        self.shrink = SoftShrink(r3)

        # Décodeur miroir (ConvTranspose = généralisation apprise de U, V)
        self.up5 = nn.ConvTranspose2d(r3, 96, kernel_size=5, stride=5)
        self.dec2 = conv_block(96 + 96, 96)
        self.up3 = nn.ConvTranspose2d(96, 64, kernel_size=3, stride=3)
        self.dec1 = conv_block(64 + 64, 64)
        self.up2 = nn.ConvTranspose2d(64, 48, kernel_size=2, stride=2)
        self.dec0 = conv_block(48 + 48, 48)
        self.head = nn.Sequential(nn.Conv2d(48, C_bands, 3, padding=1), nn.Sigmoid())

    def bufs(self):
        return {k: getattr(self, k) for k in
                ["Ph_20", "Pw_20", "Ph_60", "Pw_60", "Ph_300", "Pw_300",
                 "R_10", "R_20", "R_60", "R_300"]}

    def forward(self, y10, y20, y60, y300):
        # Encodage natif — aucune donnée n'est rééchantillonnée hors du réseau
        F0 = self.stem10(y10)                       # (B, 48, H,    W)
        F1 = self.fuse1(self.down0(F0), self.stem20(y20))    # (B, 64, H/2)
        F2 = self.fuse2(self.down1(F1), self.stem60(y60))    # (B, 96, H/6)
        F3 = self.fuse3(self.down2(F2), self.stem300(y300))  # (B, 128, H/30)

        pre_core = self.core_head(F3)               # avant seuillage (gradient vivant)
        G = self.shrink(pre_core)                   # (B, R3, H/30, W/30), zéros exacts

        # Dropout de canaux sur les skips (entraînement) : empêche le réseau
        # de contourner G en routant tout par la géométrie S2.
        p = self.skip_dropout if self.training else 0.0
        S0 = F.dropout2d(F0, p, self.training)
        S1 = F.dropout2d(F1, p, self.training)
        S2 = F.dropout2d(F2, p, self.training)

        x = self.dec2(torch.cat([self.up5(G), S2], dim=1))
        x = self.dec1(torch.cat([self.up3(x), S1], dim=1))
        x = self.dec0(torch.cat([self.up2(x), S0], dim=1))
        S_pred = self.head(x)                       # (B, C, H, W)
        return S_pred.permute(0, 2, 3, 1), G, pre_core


# ── Entraînement ──────────────────────────────────────────────────────────────

def run_msnljtae(S, obs, ops, r3=64, epochs=1500, lr=2e-3,
                 lam_sam=0.1, lam_sparse=1e-6, supervised=False,
                 patience=250, log_every=100, verbose=True):
    H, W, C = S.shape

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32)

    inputs = [to_t(obs[k]).permute(2, 0, 1).unsqueeze(0).to(device)
              for k in ["Y10", "Y20", "Y60", "Y300"]]
    targets = {k: to_t(obs[k]).unsqueeze(0).to(device) for k in obs}
    S_target = to_t(S).unsqueeze(0).to(device) if supervised else None

    model = MSNLJTAE(C, ops, r3=r3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  MS-NL-JTAE : {n_params/1e6:.2f} M paramètres | R3={r3} | "
              f"cœur {H//30}×{W//30}×{r3} | supervisé={supervised}")

    # Weight decay uniquement sur les poids de convolution (ndim > 1) :
    # les seuils du SoftShrink, les biais et les paramètres BN en sont exclus.
    theta_id = id(model.shrink.theta)
    decay = [p for p in model.parameters() if p.ndim > 1 and id(p) != theta_id]
    no_decay = [p for p in model.parameters() if p.ndim <= 1] + [model.shrink.theta]
    opt = torch.optim.Adam([
        {"params": decay, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=30)

    warmup = 150  # époques sans seuillage : le cœur apprend avant l'élagage
    best_loss, best_S, best_G, stall = float('inf'), None, None, 0
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.shrink.enabled = ep > warmup
        if ep == warmup + 1:
            # Repartir de zéro pour la sélection : l'état retenu doit être
            # post-seuillage (cœur réellement parcimonieux).
            best_loss, stall = float('inf'), 0
        model.train()
        opt.zero_grad()
        S_pred, G, pre_core = model(*inputs)
        p10, p20, p60, p300 = torch_degrade_all(S_pred, model.bufs())

        # Pénalité L1 sur le PRÉ-cœur : gradient vivant même dans la zone
        # morte du seuillage (le G post-seuillage a un gradient nul sous tau).
        loss = (F.mse_loss(p10, targets["Y10"]) + F.mse_loss(p20, targets["Y20"])
                + F.mse_loss(p60, targets["Y60"]) + 2.0 * F.mse_loss(p300, targets["Y300"])
                + lam_sam * sam_loss(p300, targets["Y300"])
                + lam_sparse * pre_core.abs().mean())
        if supervised:
            loss = loss + F.mse_loss(S_pred, S_target) + 0.1 * sam_loss(S_pred, S_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss.item())

        # Sélection du meilleur état SUR LA PERTE (jamais sur les métriques GT).
        # Capture par une passe PROPRE : sans dropout de skips (sinon la
        # reconstruction sauvegardée est bruitée par le dropout).
        if loss.item() < best_loss - 1e-8:
            best_loss, stall = loss.item(), 0
            with torch.no_grad():
                p_save = model.skip_dropout
                model.skip_dropout = 0.0
                S_clean, G_clean, _ = model(*inputs)
                model.skip_dropout = p_save
            best_S = S_clean.squeeze(0).cpu().numpy()
            best_G = G_clean.cpu()
        else:
            stall += 1
            if stall >= patience:
                if verbose:
                    print(f"  [early stopping] époque {ep}")
                break

        if verbose and (ep % log_every == 0 or ep == 1):
            with torch.no_grad():
                sp = core_sparsity(G)
            history.append((ep, loss.item(), sp))
            print(f"  ep {ep:04d} | loss {loss.item():.3e} | sparsité G {sp:.1f}%")

    elapsed = time.time() - t0
    return np.clip(best_S, 0, 1), best_G, elapsed, history, model


if __name__ == "__main__":
    import argparse, json, os
    p = argparse.ArgumentParser(description="NL-JTAE multi-input natif (v4, proposé)")
    p.add_argument('--r3', type=int, default=64)
    p.add_argument('--epochs', type=int, default=1500)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--lam_sam', type=float, default=0.1)
    p.add_argument('--lam_sparse', type=float, default=1e-6)
    p.add_argument('--supervised', action='store_true',
                   help="mode oracle (borne supérieure, étiqueté comme tel)")
    args = p.parse_args()

    S = load_paviau()
    obs, ops = simulate_streams(S)
    S_hat, G, elapsed, hist, model = run_msnljtae(
        S, obs, ops, r3=args.r3, epochs=args.epochs, lr=args.lr,
        lam_sam=args.lam_sam, lam_sparse=args.lam_sparse, supervised=args.supervised)

    m = evaluate_all(S, S_hat, G)
    m["time_s"] = elapsed
    tag = "oracle" if args.supervised else "selfsup"
    print("\n" + format_metrics(f"v4 MS-NL-JTAE ({tag})", m))
    os.makedirs("results", exist_ok=True)
    with open(f"results/paper_v4_{tag}_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    torch.save(model.state_dict(), f"results/paper_v4_{tag}_weights.pth")
