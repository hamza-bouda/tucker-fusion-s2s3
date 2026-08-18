"""
v3_unsupervised_paviau.py
─────────────────────────────────────────────────────────────────────────────
Évaluation du modèle NL-JTAE v3 en mode NON-SUPERVISÉ sur PaviaU.

Différence clé avec run_experiment() du fichier v3 :
  - loss_super (MSE vs vérité terrain)  → SUPPRIMÉ
  - loss_SAM   (SAM vs vérité terrain)  → SUPPRIMÉ
  - Seules les pertes physiques sont utilisées :
      loss = loss_S2 + 2.0 * loss_S3 + 1e-6 * loss_sparse
  - La vérité terrain S est utilisée UNIQUEMENT pour calculer les métriques
    finales (PSNR, SAM, ERGAS, SSIM, UIQI), jamais pendant l'entraînement.

Cela simule exactement ce qu'on fait sur les données réelles d'Andorre.
─────────────────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os, sys, time
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from v3_non_lineaire_conjoint_nljtae import (
    NonLinearJointTuckerAE,
    load_dataset, simulate_observations,
    calculate_psnr, calculate_sam, calculate_ergas,
    device
)

try:
    from skimage.metrics import structural_similarity as ssim_func
    from sewar.full_ref import uqi
    HAS_SEWAR = True
except ImportError:
    HAS_SEWAR = False

def calculate_ssim(ref, fused):
    if ssim_func is None:
        return 0.0
    scores = []
    for b in range(ref.shape[2]):
        s = ssim_func(ref[..., b], fused[..., b], data_range=1.0)
        scores.append(s)
    return float(np.mean(scores))

def calculate_uiqi(ref, fused):
    if not HAS_SEWAR:
        return 0.0
    scores = []
    for b in range(ref.shape[2]):
        r = ref[..., b]
        f = np.clip(fused[..., b], 0, 1)
        scores.append(float(uqi(r, f)))
    return float(np.mean(scores))

def run_unsupervised(dataset_name="PaviaU", ranks=(16, 16, 16),
                     epochs=4000, lr=2e-3):

    print("\n" + "="*80)
    print(f"  NL-JTAE v3 — MODE NON-SUPERVISÉ — {dataset_name} | ranks={ranks}")
    print("="*80)

    # ── 1. Données ──────────────────────────────────────────────────────────
    S = load_dataset(dataset_name, crop_size=(240, 240))
    H, W, C = S.shape
    print(f"  Image vérité terrain : {H}×{W}×{C}")

    (Y_S2_10m, Y_S2_20m, Y_S2_60m, Y_S3,
     P_h_S3, P_w_S3, P_h_S2_20, P_w_S2_20, P_h_S2_60, P_w_S2_60,
     R_S3, R_S2_10, R_S2_20, R_S2_60) = simulate_observations(S)

    # ── 2. Tenseurs d'entrée ────────────────────────────────────────────────
    Y_S2_10m_t = torch.tensor(Y_S2_10m, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    Y_S2_20m_t = torch.tensor(Y_S2_20m, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    Y_S2_20m_up = nn.functional.interpolate(Y_S2_20m_t, size=(H,W), mode='bilinear', align_corners=False)
    Y_S2_60m_t = torch.tensor(Y_S2_60m, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    Y_S2_60m_up = nn.functional.interpolate(Y_S2_60m_t, size=(H,W), mode='bilinear', align_corners=False)
    Y_S2_up  = torch.cat([Y_S2_10m_t, Y_S2_20m_up, Y_S2_60m_up], dim=1).to(device)
    Y_S3_t   = torch.tensor(Y_S3, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    Y_S3_up  = nn.functional.interpolate(Y_S3_t, size=(H,W), mode='bilinear', align_corners=False).to(device)
    X_joint  = torch.cat([Y_S2_up, Y_S3_up], dim=1)  # (1,34,240,240)

    # Cibles physiques (S2 et S3 dégradés) — PAS la vérité terrain S
    Y_S2_10m_target = Y_S2_10m_t.to(device)                        # (1,4,240,240)
    Y_S2_20m_target = Y_S2_20m_t.permute(0,2,3,1).to(device)       # (1,120,120,6)
    Y_S2_60m_target = Y_S2_60m_t.permute(0,2,3,1).to(device)       # (1,40,40,3)
    Y_S3_target     = Y_S3_t.permute(0,2,3,1).to(device)            # (1,8,8,C)

    # ── 3. Modèle ───────────────────────────────────────────────────────────
    model = NonLinearJointTuckerAE(
        H, W, C, ranks,
        P_h_S3, P_w_S3, P_h_S2_20, P_w_S2_20, P_h_S2_60, P_w_S2_60,
        R_S3, R_S2_10, R_S2_20, R_S2_60
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)

    best_loss  = float('inf')
    best_S     = None
    no_improve = 0
    patience   = 200

    print("\n  Entraînement NON-SUPERVISÉ (sans vérité terrain dans la loss)...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        S_pred, y_S2_pred_full, y_S3_pred, G = model(X_joint)

        # ── Pertes physiques uniquement (identique à Andorre) ──────────────
        # Perte S2 10m
        loss_S2_10m = torch.mean((y_S2_pred_full[:,:,:,:4].permute(0,3,1,2)
                                  - Y_S2_10m_target) ** 2)

        # Perte S2 20m
        y20 = y_S2_pred_full[:,:,:,4:10]
        y20t = torch.einsum('ij,bcjl->bcil', model.P_h_S2_20, y20.permute(0,3,1,2))
        y20s = torch.einsum('bcil,kl->bcik', y20t, model.P_w_S2_20).permute(0,2,3,1)
        loss_S2_20m = torch.mean((y20s - Y_S2_20m_target) ** 2)

        # Perte S3
        loss_S3 = torch.mean((y_S3_pred - Y_S3_target) ** 2)

        # Régularisation L1 sur G
        loss_sparse = torch.mean(torch.abs(G))

        # Pas de loss_super, pas de loss_SAM !
        loss = loss_S2_10m + loss_S2_20m + 2.0 * loss_S3 + 1e-6 * loss_sparse

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss - 1e-7:
            best_loss  = loss.item()
            no_improve = 0
            best_S     = S_pred.squeeze(0).detach().cpu().numpy()
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [Early Stopping] époque {epoch}")
                break

        if epoch % 100 == 0 or epoch == 1:
            sparsity = float((torch.abs(G) < 1e-4).float().mean()) * 100
            print(f"  Epoch {epoch:04d}/{epochs} | Loss={loss.item():.3e} "
                  f"(S2={loss_S2_10m.item()+loss_S2_20m.item():.3e}, "
                  f"S3={loss_S3.item():.3e}) | Sparsité G={sparsity:.1f}%")

    print(f"\n  Entraînement terminé en {time.time()-t0:.1f}s")

    # ── 4. Sauvegarder best_S et S pour calcul local ─────────────────────────
    best_S = np.clip(best_S, 0, 1)
    np.save("results/unsup_best_S.npy",  best_S)
    np.save("results/unsup_gt_S.npy",    S)
    print("  Fichiers sauvegardés : results/unsup_best_S.npy, results/unsup_gt_S.npy")

    # ── 5. Métriques avec skimage uniquement ─────────────────────────────────
    print("\n  Calcul des métriques contre la vérité terrain PaviaU...")

    psnr  = calculate_psnr(S, best_S)
    sam   = calculate_sam(S, best_S)
    ergas = calculate_ergas(S, best_S, ratio=30)

    # SSIM bande par bande
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        ssim_scores = [ssim_fn(S[..., b], best_S[..., b], data_range=1.0)
                       for b in range(S.shape[2])]
        ssim_val = float(np.mean(ssim_scores))
    except Exception:
        ssim_val = 0.0

    sparsity_final = float((torch.abs(G) < 1e-4).float().mean()) * 100

    print("\n" + "="*60)
    print("  RÉSULTATS — NL-JTAE v3 NON-SUPERVISÉ sur PaviaU")
    print("="*60)
    print(f"  PSNR  : {psnr:.2f} dB")
    print(f"  SAM   : {sam:.2f}°")
    print(f"  ERGAS : {ergas:.4f}")
    print(f"  SSIM  : {ssim_val:.4f}")
    print(f"  Sparsité G : {sparsity_final:.1f}%")
    print("="*60)
    print("\n  RAPPEL — NL-JTAE v3 SUPERVISÉ sur PaviaU :")
    print("  PSNR=40.57 dB | SAM=1.55° | ERGAS=0.1483 | SSIM=0.9862")
    print("="*60)

    return psnr, sam, ergas, ssim_val

if __name__ == "__main__":
    run_unsupervised(dataset_name="PaviaU", ranks=(16, 16, 16), epochs=4000, lr=2e-3)
