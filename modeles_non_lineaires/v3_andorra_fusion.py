import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from modeles_lineaires.v0_lineaire_math_utils import estimer_matrices_degradation, create_spatial_degradation_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de l'appareil : {device}")

def main():
    data_dir = "./data/andorra/20220709"
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # 1. CHARGEMENT ET CROP DES DONNÉES D'ANDORRE
    # --------------------------------------------------------------------------
    print("\n[1/5] Chargement et normalisation des images d'Andorre...")
    s2_10m_full = tiff.imread(os.path.join(data_dir, "31TCH_20220709_8800_8800_target_s2_10m.tif"))
    s2_20m_full = tiff.imread(os.path.join(data_dir, "31TCH_20220709_8800_8800_target_s2_20m.tif"))
    s3_full = tiff.imread(os.path.join(data_dir, "31TCH_20220709_8800_8800_target_s3.tif"))
    
    # Coordonnées du crop propre sans NaNs ni zéros
    y_10, x_10 = 720, 0
    y_20, x_20 = y_10 // 2, x_10 // 2
    
    # Extraction des crops
    M1 = s2_10m_full[y_10:y_10+480, x_10:x_10+480].astype(np.float32)
    M2 = s2_20m_full[y_20:y_20+240, x_20:x_20+240].astype(np.float32)
    H_n_full = s3_full[y_10:y_10+480, x_10:x_10+480].astype(np.float32)
    
    # Décimer S3 par un facteur de 30 pour obtenir la basse résolution physique
    H_n = H_n_full[::30, ::30, :]  # Shape (16, 16, 17)
    
    # Normalisation Min/Max par bande
    def normalize_bandwise(img):
        H, W, C = img.shape
        img_norm = np.zeros_like(img)
        img_min = np.zeros(C)
        img_max = np.zeros(C)
        for c in range(C):
            c_min = img[..., c].min()
            c_max = img[..., c].max()
            img_min[c] = c_min
            img_max[c] = c_max
            if c_max - c_min == 0:
                img_norm[..., c] = 0.0
            else:
                img_norm[..., c] = (img[..., c] - c_min) / (c_max - c_min)
        return img_norm, img_min, img_max
        
    M1_norm, M1_min, M1_max = normalize_bandwise(M1)
    M2_norm, M2_min, M2_max = normalize_bandwise(M2)
    H_n_norm, H_n_min, H_n_max = normalize_bandwise(H_n)
    
    H_dim, W_dim, C_S2_10 = M1_norm.shape
    C_S3 = H_n_norm.shape[2]
    
    print(f"  - S2 10m shape : {M1_norm.shape}")
    print(f"  - S2 20m shape : {M2_norm.shape}")
    print(f"  - S3 300m shape: {H_n_norm.shape}")
    
    # --------------------------------------------------------------------------
    # 2. DEFINITION DES MATRICES DE DEGRADATION PHYSIQUE
    # --------------------------------------------------------------------------
    print("\n[2/5] Initialisation des matrices physiques (PSF et SRF)...")
    # Pour la PSF spatiale, on utilise une valeur de sigma standard basée sur le facteur d'échelle
    sigma_10 = 30.0 / 2.355
    sigma_20 = 15.0 / 2.355
    
    # R_S2_10: 4x21 (S2 B2, B3, B4, B8 -> S3 Oa4, Oa6, Oa8, Oa17)
    # Longueurs d'ondes : B2=490nm (Oa4), B3=560nm (Oa6), B4=665nm (Oa8), B8=842nm (Oa17)
    R_S2_10 = np.zeros((4, 21))
    R_S2_10[0, 3] = 1.0   # S2 B2 (index 0) -> S3 Oa4 (index 3)
    R_S2_10[1, 5] = 1.0   # S2 B3 (index 1) -> S3 Oa6 (index 5)
    R_S2_10[2, 7] = 1.0   # S2 B4 (index 2) -> S3 Oa8 (index 7)
    R_S2_10[3, 16] = 1.0  # S2 B8 (index 3) -> S3 Oa17 (index 16)
    
    # R_S2_20: 6x21 (S2 B5, B6, B7, B8a, B11, B12 -> S3 Oa11, Oa12, Oa16, Oa17, SWIR=aucun)
    R_S2_20 = np.zeros((6, 21))
    R_S2_20[0, 10] = 1.0  # B5 (705nm) -> Oa11 (709nm, index 10)
    R_S2_20[1, 11] = 1.0  # B6 (740nm) -> Oa12 (753nm, index 11)
    R_S2_20[2, 15] = 1.0  # B7 (783nm) -> Oa16 (779nm, index 15)
    R_S2_20[3, 16] = 1.0  # B8a (865nm) -> Oa17 (865nm, index 16)
    # B11 et B12 (SWIR) n'ont pas d'équivalent dans S3 OLCI (VNIR), donc restent à 0
    
    # R_S2_60: 3x21 (complètement à zéro)
    R_S2_60 = np.zeros((3, 21))
    
    # Sentinel-3 : On veut reconstruire 21 bandes, donc R_S3 est l'identité 21x21
    R_S3 = np.eye(21)
    
    # Matrices spatiales
    P_h_S3 = create_spatial_degradation_matrix(H_dim, scale=30, sigma=sigma_10)
    P_w_S3 = create_spatial_degradation_matrix(W_dim, scale=30, sigma=sigma_10)
    
    P_h_S2_20 = create_spatial_degradation_matrix(H_dim, scale=2, sigma=sigma_20)
    P_w_S2_20 = create_spatial_degradation_matrix(W_dim, scale=2, sigma=sigma_20)
    
    P_h_S2_60 = np.zeros((80, H_dim))
    P_w_S2_60 = np.zeros((80, W_dim))
    
    # --------------------------------------------------------------------------
    # 3. PREPARATION DES ENTRÉES ET DU MODÈLE
    # --------------------------------------------------------------------------
    print("\n[3/5] Préparation du tenseur d'entrée rembourré à 34 canaux...")
    
    # Upsampling bilinéaire des bandes de S2 et S3 pour l'entrée
    def upsample_image(img, target_shape):
        tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0) # (1, C, h, w)
        up_tensor = nn.functional.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=True)
        return up_tensor.squeeze(0).permute(1, 2, 0).numpy()
        
    M2_up = upsample_image(M2_norm, (H_dim, W_dim))
    H_n_up = upsample_image(H_n_norm, (H_dim, W_dim))
    
    # Rembourrage S2 : 4 + 6 = 10 bands -> 13 bands
    S2_joint = np.concatenate([M1_norm, M2_up], axis=2)
    S2_joint_padded = np.concatenate([S2_joint, np.zeros((H_dim, W_dim, 3))], axis=2)
    
    # Rembourrage S3 : 17 bands -> 21 bands
    H_n_up_padded = np.concatenate([H_n_up, np.zeros((H_dim, W_dim, 4))], axis=2)
    
    # Concaténation finale = 13 + 21 = 34 bandes
    x_joint = np.concatenate([S2_joint_padded, H_n_up_padded], axis=2)
    x_joint_tensor = torch.tensor(x_joint, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) # (1, 34, 240, 240)
    
    # Instanciation du modèle NL-JTAE v3
    ranks = (16, 16, 16)
    model = NonLinearJointTuckerAE(
        H_dim, W_dim, 21, ranks,  # 21 bandes de sortie dans le modèle
        P_h_S3, P_w_S3, P_h_S2_20, P_w_S2_20, P_h_S2_60, P_w_S2_60,
        R_S3, R_S2_10, R_S2_20, R_S2_60
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    
    # Cibles (avec S3 cible rembourrée à 21 bandes)
    Y_S2_10m_target = torch.tensor(M1_norm, dtype=torch.float32).to(device)
    Y_S2_20m_target = torch.tensor(M2_norm, dtype=torch.float32).to(device)
    
    H_n_target_padded = np.concatenate([H_n_norm, np.zeros((H_n_norm.shape[0], H_n_norm.shape[1], 4))], axis=2)
    Y_S3_target = torch.tensor(H_n_target_padded, dtype=torch.float32).to(device)
    
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
        
        # S_pred: (1, 240, 240, 17)
        S_pred, y_S2_pred_full, y_S3_pred, G = model(x_joint_tensor)
        
        # 1. Perte S2 10m (canaux 0:4)
        loss_S2_10m = torch.mean((y_S2_pred_full[:, :, :, 0:4] - Y_S2_10m_target) ** 2)
        
        # 2. Perte S2 20m (canaux 4:10)
        y_S2_pred_20m_full = y_S2_pred_full[:, :, :, 4:10]
        y_S2_pred_20m_temp = torch.einsum('ij,bcjl->bcil', model.P_h_S2_20, y_S2_pred_20m_full.permute(0, 3, 1, 2))
        y_S2_pred_20m = torch.einsum('bcil,kl->bcik', y_S2_pred_20m_temp, model.P_w_S2_20).permute(0, 2, 3, 1)
        loss_S2_20m = torch.mean((y_S2_pred_20m - Y_S2_20m_target) ** 2)
        
        # Pas de perte 60m
        loss_S2 = loss_S2_10m + loss_S2_20m
        
        # 3. Perte S3
        loss_S3 = torch.mean((y_S3_pred - Y_S3_target) ** 2)
        
        # 4. Régularisation L1 sur G
        loss_sparse = torch.mean(torch.abs(G))
        
        # Perte totale
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
    # Création d'une figure de comparaison en vraies couleurs (RGB)
    # Sentinel-3 OLCI Andorre RGB : R=index 7, G=index 5, B=index 3
    rgb_bands = [7, 5, 3]
    
    # Extraire les bandes RGB normalisées de la prédiction avant dénormalisation pour l'affichage
    best_S_display = best_S[..., rgb_bands]
    
    # Dénormalisation bande par bande pour retrouver l'échelle de réflectance physique d'origine
    for c in range(17):
        best_S[..., c] = best_S[..., c] * (H_n_max[c] - H_n_min[c]) + H_n_min[c]
        
    # Découper pour ne conserver que les 17 bandes spectrales réelles d'Andorre
    best_S = best_S[..., :17]
        
    tiff_path = os.path.join(results_dir, "andorra_super_resolved_v3.tif")
    tiff.imwrite(tiff_path, best_S.astype(np.float32))
    print(f"Image super-résolue Andorre (17 bandes, 10m) sauvegardée : {tiff_path}")
    
    weights_path = os.path.join(results_dir, "andorra_NL_JTAE_v3_weights.pth")
    torch.save(best_model_state, weights_path)
    print(f"Poids du modèle sauvegardés sous : {weights_path}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')
    
    for ax in axes:
        ax.axis('off')
        
    # Sentinel-2 (Rouge=2, Vert=1, Bleu=0)
    axes[0].imshow(M1_norm[..., [2, 1, 0]])
    axes[0].set_title("Sentinel-2 Input (10m Spatial RGB)", color='white')
    
    # Sentinel-3 d'origine upsamplé
    s3_up_display = H_n_up[..., rgb_bands]
    axes[1].imshow(s3_up_display)
    axes[1].set_title("Sentinel-3 Input (300m Spectral RGB)", color='white')
    
    # Sentinel-3 super-résolue à 10m
    axes[2].imshow(best_S_display)
    axes[2].set_title("Reconstructed S3 at 10m (NL-JTAE v3)", color='white')
    
    plt.suptitle("Fusion Réelle Sentinel-2 / Sentinel-3 (Andorre - Juillet 2022) - Modèle NL-JTAE v3", color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = "./v3_andorra_fusion_comparaison.png"
    plt.savefig(plot_path, dpi=150, facecolor='#0d1117')
    print(f"Figure sauvegardée sous : {plot_path}")
    
    # Copier dans les artéfacts
    artifact_dest = "C:\\Users\\hamza\\.gemini\\antigravity\\brain\\15201f33-80da-4d2c-a3f4-2b3479314927\\v3_andorra_fusion_comparaison.png"
    import shutil
    shutil.copy(plot_path, artifact_dest)
    print("Figure copiée dans les artéfacts.")

if __name__ == "__main__":
    main()
