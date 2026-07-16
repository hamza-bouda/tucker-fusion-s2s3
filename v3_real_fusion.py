import os
import time
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Importer les modules locaux
from v3_non_lineaire_conjoint_nljtae import NonLinearJointTuckerAE
from v0_lineaire_math_utils import estimer_matrices_degradation, create_spatial_degradation_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de l'appareil : {device}")

def main():
    data_dir = "./data"
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # 1. CHARGEMENT DES CROPS REELS (FEVRIER 2023)
    # --------------------------------------------------------------------------
    print("\n[1/5] Chargement et normalisation des crops réels...")
    s2_10m_full = tiff.imread(os.path.join(data_dir, "s2_10m_2023-09.tif"))
    s2_20m_full = tiff.imread(os.path.join(data_dir, "s2_20m_2023-09.tif"))
    s2_60m_full = tiff.imread(os.path.join(data_dir, "s2_60m_2023-09.tif"))
    s3_300m_full = tiff.imread(os.path.join(data_dir, "s3_300m_2023-09.tif"))
    
    # Correction spectrale des bandes Sentinel-3 corrompues dans le TIFF (remplacement des zéros et pics)
    s3_300m_full[..., 0] = s3_300m_full[..., 1]
    s3_300m_full[..., 9] = 0.5 * (s3_300m_full[..., 8] + s3_300m_full[..., 10])
    
    # Coordonnées du crop propre sans NaNs trouvées précédemment
    y1_10, x1_10 = 1211, 443
    y2_10, x2_10 = y1_10 + 240, x1_10 + 240
    
    y1_20, x1_20 = y1_10 // 2, x1_10 // 2
    y2_20, x2_20 = y1_20 + 120, x1_20 + 120
    
    y1_60, x1_60 = y1_10 // 6, x1_10 // 6
    y2_60, x2_60 = y1_60 + 40, x1_60 + 40
    
    y1_300, x1_300 = y1_10 // 30, x1_10 // 30
    y2_300, x2_300 = y1_300 + 8, x1_300 + 8
    
    # Extraction des patches
    M1 = s2_10m_full[y1_10:y2_10, x1_10:x2_10].astype(np.float32)
    M2 = s2_20m_full[y1_20:y2_20, x1_20:x2_20].astype(np.float32)
    M3 = s2_60m_full[y1_60:y2_60, x1_60:x2_60].astype(np.float32)
    H_n = s3_300m_full[y1_300:y2_300, x1_300:x2_300].astype(np.float32)
    
    # Normalisation Min/Max à [0, 1] pour chaque patch pour stabiliser les réseaux
    def normalize(img):
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min == 0:
            return np.zeros_like(img)
        return (img - img_min) / (img_max - img_min)
        
    M1_norm = normalize(M1)
    M2_norm = normalize(M2)
    M3_norm = normalize(M3)
    H_n_norm = normalize(H_n)
    
    H_dim, W_dim, C_S2_10 = M1_norm.shape
    C_S3 = H_n_norm.shape[2]
    
    print(f"  - S2 10m shape : {M1_norm.shape}")
    print(f"  - S2 20m shape : {M2_norm.shape}")
    print(f"  - S2 60m shape : {M3_norm.shape}")
    print(f"  - S3 300m shape: {H_n_norm.shape}")
    
    # --------------------------------------------------------------------------
    # 2. ESTIMATION DES MATRICES DE DEGRADATION PHYSIQUE
    # --------------------------------------------------------------------------
    print("\n[2/5] Estimation des matrices physiques (PSF et SRF)...")
    sigma_10, R_S2_10 = estimer_matrices_degradation(M1_norm, H_n_norm, facteur_echelle=30)
    sigma_20, R_S2_20 = estimer_matrices_degradation(M2_norm, H_n_norm, facteur_echelle=15)
    sigma_60, R_S2_60 = estimer_matrices_degradation(M3_norm, H_n_norm, facteur_echelle=5)
    
    # Matrices spatiales
    P_h_S3 = create_spatial_degradation_matrix(H_dim, scale=30, sigma=sigma_10)
    P_w_S3 = create_spatial_degradation_matrix(W_dim, scale=30, sigma=sigma_10)
    
    P_h_S2_20 = create_spatial_degradation_matrix(H_dim, scale=2, sigma=sigma_10/2.0)
    P_w_S2_20 = create_spatial_degradation_matrix(W_dim, scale=2, sigma=sigma_10/2.0)
    
    P_h_S2_60 = create_spatial_degradation_matrix(H_dim, scale=6, sigma=sigma_10/6.0)
    P_w_S2_60 = create_spatial_degradation_matrix(W_dim, scale=6, sigma=sigma_10/6.0)
    
    # Sentinel-3 : On veut reconstruire ses 21 bandes, donc R_S3 est l'identité
    R_S3 = np.eye(C_S3)
    
    # --------------------------------------------------------------------------
    # 3. PREPARATION DES ENTRÉES ET DU MODÈLE
    # --------------------------------------------------------------------------
    print("\n[3/5] Préparation du tenseur d'entrée 34 canaux...")
    
    # Upsampling bilinéaire des bandes de S2 et S3 pour l'entrée
    def upsample_image(img, target_shape):
        tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0) # (1, C, h, w)
        up_tensor = nn.functional.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=True)
        return up_tensor.squeeze(0).permute(1, 2, 0).numpy()
        
    M2_up = upsample_image(M2_norm, (H_dim, W_dim))
    M3_up = upsample_image(M3_norm, (H_dim, W_dim))
    H_n_up = upsample_image(H_n_norm, (H_dim, W_dim))
    
    # Concaténation de Sentinel-2 (13 bandes) et Sentinel-3 (21 bandes) = 34 bandes
    S2_joint = np.concatenate([M1_norm, M2_up, M3_up], axis=2)
    x_joint = np.concatenate([S2_joint, H_n_up], axis=2)
    x_joint_tensor = torch.tensor(x_joint, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 34, 240, 240)
    
    # Instanciation du modèle NL-JTAE v3
    ranks = (16, 16, 16) # R3=16 car on a 21 bandes de sortie
    model = NonLinearJointTuckerAE(
        H_dim, W_dim, C_S3, ranks,
        P_h_S3, P_w_S3, P_h_S2_20, P_w_S2_20, P_h_S2_60, P_w_S2_60,
        R_S3, R_S2_10, R_S2_20, R_S2_60
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    
    # Cibles pour l'apprentissage non-supervisé (physique uniquement)
    Y_S2_10m_target = torch.tensor(M1_norm, dtype=torch.float32).to(device)
    Y_S2_20m_target = torch.tensor(M2_norm, dtype=torch.float32).to(device)
    Y_S2_60m_target = torch.tensor(M3_norm, dtype=torch.float32).to(device)
    Y_S3_target = torch.tensor(H_n_norm, dtype=torch.float32).to(device)
    
    # --------------------------------------------------------------------------
    # 4. ENTRAÎNEMENT SANS VÉRITÉ TERRAIN (NON-SUPERVISÉ)
    # --------------------------------------------------------------------------
    print("\n[4/5] Début de l'apprentissage physique non-supervisé...")
    epochs = 4000
    patience = 200
    best_loss = float('inf')
    best_S = None
    epochs_no_improve = 0
    
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # S_pred: (1, 240, 240, 21)
        S_pred, y_S2_pred_full, y_S3_pred, G = model(x_joint_tensor)
        
        # 1. Perte S2 10m
        loss_S2_10m = torch.mean((y_S2_pred_full[:, :, :, 0:4] - Y_S2_10m_target) ** 2)
        # 2. Perte S2 20m
        y_S2_pred_20m_full = y_S2_pred_full[:, :, :, 4:10]
        y_S2_pred_20m_temp = torch.einsum('ij,bcjl->bcil', model.P_h_S2_20, y_S2_pred_20m_full.permute(0, 3, 1, 2))
        y_S2_pred_20m = torch.einsum('bcil,kl->bcik', y_S2_pred_20m_temp, model.P_w_S2_20).permute(0, 2, 3, 1)
        loss_S2_20m = torch.mean((y_S2_pred_20m - Y_S2_20m_target) ** 2)
        # 3. Perte S2 60m
        y_S2_pred_60m_full = y_S2_pred_full[:, :, :, 10:13]
        y_S2_pred_60m_temp = torch.einsum('ij,bcjl->bcil', model.P_h_S2_60, y_S2_pred_60m_full.permute(0, 3, 1, 2))
        y_S2_pred_60m = torch.einsum('bcil,kl->bcik', y_S2_pred_60m_temp, model.P_w_S2_60).permute(0, 2, 3, 1)
        loss_S2_60m = torch.mean((y_S2_pred_60m - Y_S2_60m_target) ** 2)
        loss_S2 = loss_S2_10m + loss_S2_20m + loss_S2_60m
        
        # 4. Perte S3
        loss_S3 = torch.mean((y_S3_pred - Y_S3_target) ** 2)
        
        # 5. Régularisation L1 sur G
        loss_sparse = torch.mean(torch.abs(G))
        
        # Perte physique totale (pas de perte super-résolue ou SAM directe)
        loss = loss_S2 + 2.0 * loss_S3 + 1e-6 * loss_sparse
        
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss)
        
        sparsity = float((torch.abs(G) < 1e-4).float().mean().item()) * 100
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.2e} (S2: {loss_S2.item():.2e}, S3: {loss_S3.item():.2e}) | G Sparsity: {sparsity:.1f}%")
            
        if loss.item() < best_loss - 1e-7:
            best_loss = loss.item()
            epochs_no_improve = 0
            best_S = S_pred.squeeze(0).detach().cpu().numpy()
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] Déclenché à l'époque {epoch}")
                break
                
    print(f"\nEntraînement complété en {time.time()-t_start:.2f}s.")
    
    # --------------------------------------------------------------------------
    # 5. SAUVEGARDE ET VISUALISATION
    # --------------------------------------------------------------------------
    print("\n[5/5] Sauvegarde des résultats...")
    tiff_path = os.path.join(results_dir, "real_super_resolved_v3.tif")
    tiff.imwrite(tiff_path, best_S.astype(np.float32))
    print(f"Image super-résolue réelle (21 bandes, 10m) sauvegardée : {tiff_path}")
    
    # Sauvegarde des poids du modèle
    weights_path = os.path.join(results_dir, "real_NL_JTAE_v3_weights.pth")
    torch.save(best_model_state, weights_path)
    print(f"Poids du modèle sauvegardés sous : {weights_path}")
    
    # Création d'une figure de comparaison en vraies couleurs (RGB)
    # Bandes Sentinel-3 : Rouge=Bande 8 (index 7), Vert=Bande 6 (index 5), Bleu=Bande 4 (index 3)
    rgb_bands = [7, 5, 3]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')
    
    for ax in axes:
        ax.axis('off')
        
    # Sentinel-2 (Bandes 10m d'origine en RGB : Rouge=Index 2, Vert=Index 1, Bleu=Index 0)
    axes[0].imshow(M1_norm[..., [2, 1, 0]])
    axes[0].set_title("Sentinel-2 Input (10m Spatial RGB)", color='white')
    
    # Sentinel-3 d'origine (Basse résolution, interpolée pour l'affichage en RGB)
    s3_up_display = H_n_up[..., rgb_bands]
    axes[1].imshow(s3_up_display / (s3_up_display.max() + 1e-8))
    axes[1].set_title("Sentinel-3 Input (300m Spectral RGB)", color='white')
    
    # Sentinel-3 super-résolue à 10m (Reconstruction par notre modèle v3)
    s3_fused_display = best_S[..., rgb_bands]
    axes[2].imshow(s3_fused_display / (s3_fused_display.max() + 1e-8))
    axes[2].set_title("Reconstructed S3 at 10m (NL-JTAE v3)", color='white')
    
    plt.suptitle("Fusion Réelle Sentinel-2 / Sentinel-3 (Septembre 2023) - Modèle NL-JTAE v3", color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = "./v3_real_fusion_comparaison.png"
    plt.savefig(plot_path, dpi=150, facecolor='#0d1117')
    print(f"Figure sauvegardée sous : {plot_path}")
    
    # Copier dans les artéfacts
    artifact_dest = "C:\\Users\\hamza\\.gemini\\antigravity\\brain\\15201f33-80da-4d2c-a3f4-2b3479314927\\v3_real_fusion_comparaison.png"
    import shutil
    shutil.copy(plot_path, artifact_dest)
    print("Figure copiée dans les artéfacts.")

if __name__ == "__main__":
    main()
