import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import urllib.request
import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import tifffile as tiff
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de l'appareil : {device}")

# ── ÉTAPE 1 : CHARGEMENT DES JEUX DE DONNÉES ──────────────────────────────────

from v1_autoencodeur_couple_ctae import load_dataset, simulate_observations

# ── ÉTAPE 2 : ARCHITECTURE NON-LINÉAIRE AVEC CONVTRANSPOSE2D ET SKIP CONNECTIONS ──

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class TuckerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, out_spatial_grid, input_size=(240, 240)):
        super().__init__()
        # Etape 1 : Résolution 240x240
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.inception = InceptionBlock(64, 64)
        
        # Etape 2 : Downsampling x5 vers 48x48
        self.down_x5 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.res_conv_48 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Etape 3 : Downsampling x3 vers 16x16
        self.down_x3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.res_conv_16 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Entrée x: (B, 34, 240, 240)
        out_240 = self.init_conv(x)
        feat_240 = self.inception(out_240)
        out_240 = out_240 + feat_240 # (B, 64, 240, 240) - Skip connection 240
        
        out_48 = self.down_x5(out_240) # (B, 64, 48, 48)
        feat_48 = self.res_conv_48(out_48) # (B, 128, 48, 48) - Skip connection 48
        
        out_16 = self.down_x3(feat_48) # (B, 128, 16, 16)
        G = self.res_conv_16(out_16) # (B, out_channels, 16, 16)
        
        return G.permute(0, 2, 3, 1), feat_240, feat_48

class NonLinearTuckerDecoder(nn.Module):
    def __init__(self, out_channels=103, in_channels=8, spatial_in=(16, 16), spatial_out=(240, 240)):
        super().__init__()
        # Etape 1 : ConvTranspose x3 (16x16 -> 48x48)
        self.up_x3 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=3, padding=0)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # 128 (up) + 128 (skip) = 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Etape 2 : ConvTranspose x5 (48x48 -> 240x240)
        self.up_x5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=5, padding=0)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # 64 (up) + 64 (skip) = 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Etape 3 : Conv finale
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, G, feat_240, feat_48):
        # G: (B, R1, R2, R3) -> (B, R3, R1, R2)
        x = G.permute(0, 3, 1, 2)
        
        # Upsample x3 et concaténation avec feat_48
        x = self.up_x3(x) # (B, 128, 48, 48)
        x = torch.cat([x, feat_48], dim=1) # (B, 256, 48, 48)
        x = self.conv_block1(x) # (B, 128, 48, 48)
        
        # Upsample x5 et concaténation avec feat_240
        x = self.up_x5(x) # (B, 64, 240, 240)
        x = torch.cat([x, feat_240], dim=1) # (B, 128, 240, 240)
        x = self.conv_block2(x) # (B, 64, 240, 240)
        
        # Final conv
        out = self.final_conv(x) # (B, C_bands, 240, 240)
        return out.permute(0, 2, 3, 1) # (B, H, W, C_bands)

class NonLinearJointTuckerAE(nn.Module):
    def __init__(self, H, W, C_bands, ranks, 
                 P_h_S3, P_w_S3, P_h_S2_20, P_w_S2_20, P_h_S2_60, P_w_S2_60,
                 R_S3, R_S2_10, R_S2_20, R_S2_60):
        super().__init__()
        self.ranks = ranks
        R1, R2, R3 = ranks
        
        # Enregistrement des matrices physiques
        self.register_buffer('P_h_S3', torch.tensor(P_h_S3, dtype=torch.float32))
        self.register_buffer('P_w_S3', torch.tensor(P_w_S3, dtype=torch.float32))
        self.register_buffer('P_h_S2_20', torch.tensor(P_h_S2_20, dtype=torch.float32))
        self.register_buffer('P_w_S2_20', torch.tensor(P_w_S2_20, dtype=torch.float32))
        self.register_buffer('P_h_S2_60', torch.tensor(P_h_S2_60, dtype=torch.float32))
        self.register_buffer('P_w_S2_60', torch.tensor(P_w_S2_60, dtype=torch.float32))
        
        self.register_buffer('R_S3', torch.tensor(R_S3, dtype=torch.float32))
        self.register_buffer('R_S2_10', torch.tensor(R_S2_10, dtype=torch.float32))
        self.register_buffer('R_S2_20', torch.tensor(R_S2_20, dtype=torch.float32))
        self.register_buffer('R_S2_60', torch.tensor(R_S2_60, dtype=torch.float32))
        
        R_S2 = np.concatenate([R_S2_10, R_S2_20, R_S2_60], axis=0)
        self.register_buffer('R_S2', torch.tensor(R_S2, dtype=torch.float32))
        
        # Encodeur unique conjoint : 13 + 21 = 34 canaux
        self.encoder = TuckerEncoder(in_channels=34, out_channels=R3, out_spatial_grid=(R1, R2), input_size=(H, W))
        
        # Décodeur 100% non-linéaire avec ConvTranspose et Skips
        self.decoder = NonLinearTuckerDecoder(out_channels=C_bands, in_channels=R3, spatial_in=(R1, R2), spatial_out=(H, W))

    def forward(self, x_joint):
        # Forward pass de l'encodeur avec récupération des skip connections
        G, feat_240, feat_48 = self.encoder(x_joint)
        
        # 1. Super-résolution HSI complète
        S_pred = self.decoder(G, feat_240, feat_48)  # (B, H, W, C)
        
        # 2. Projection physique S2 (13 bandes)
        y_S2_pred_full = torch.matmul(S_pred, self.R_S2.T)
        
        # 3. Projection physique S3 (21 bandes à 300m)
        S_pred_t = S_pred.permute(0, 3, 1, 2) # (B, C, H, W)
        S_S3_temp = torch.einsum('ij,bcjl->bcil', self.P_h_S3, S_pred_t)
        S_S3_spatial = torch.einsum('bcil,kl->bcik', S_S3_temp, self.P_w_S3).permute(0, 2, 3, 1) # (B, 8, 8, C)
        y_S3_pred = torch.matmul(S_S3_spatial, self.R_S3.T)
        
        return S_pred, y_S2_pred_full, y_S3_pred, G

# ── ÉTAPE 3 : EVALUATION ET PERTES SPÉCIALISÉES ───────────────────────────────

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    ssim_func = None

def calculate_psnr(ref, fused):
    rmse_val = np.sqrt(np.mean((ref - fused) ** 2))
    if rmse_val == 0: return 100.0
    return float(20 * np.log10(1.0 / rmse_val))

def calculate_sam(ref, fused):
    n_pixels = ref.shape[0] * ref.shape[1]
    ref_flat = ref.reshape(n_pixels, ref.shape[2])
    fused_flat = fused.reshape(n_pixels, fused.shape[2])
    dot_prod = np.sum(ref_flat * fused_flat, axis=1)
    norms = np.linalg.norm(ref_flat, axis=1) * np.linalg.norm(fused_flat, axis=1)
    cos_theta = np.clip(dot_prod / (norms + 1e-8), -1.0, 1.0)
    sam_rad = np.arccos(cos_theta)
    return float(np.mean(sam_rad) * 180.0 / np.pi)

def calculate_ergas(ref, fused, ratio=30):
    h, w, n_bands = ref.shape
    rmse_bands = []
    mean_ref_bands = []
    for b in range(n_bands):
        diff = ref[..., b] - fused[..., b]
        rmse_bands.append(np.sqrt(np.mean(diff ** 2)))
        mean_ref_bands.append(np.mean(ref[..., b]))
    rmse_bands = np.array(rmse_bands)
    mean_ref_bands = np.array(mean_ref_bands)
    val = np.sum((rmse_bands / (mean_ref_bands + 1e-8)) ** 2)
    return float(100.0 / ratio * np.sqrt(val / n_bands))

def calculate_uiqi(x, y):
    n_bands = x.shape[-1]
    uiqis = []
    for b in range(n_bands):
        xb = x[..., b]
        yb = y[..., b]
        x_mean = np.mean(xb)
        y_mean = np.mean(yb)
        x_var = np.var(xb)
        y_var = np.var(yb)
        xy_cov = np.mean((xb - x_mean) * (yb - y_mean))
        
        num = 4.0 * xy_cov * x_mean * y_mean
        denom = (x_var + y_var) * (x_mean**2 + y_mean**2)
        if denom == 0:
            q = 1.0 if x_var == 0 and y_var == 0 else 0.0
        else:
            q = num / denom
        uiqis.append(q)
    return float(np.mean(uiqis))

def calculate_ssim(x, y):
    if ssim_func is None:
        return 0.0
    n_bands = x.shape[-1]
    ssims = []
    for b in range(n_bands):
        val = ssim_func(x[..., b], y[..., b], data_range=1.0)
        ssims.append(val)
    return float(np.mean(ssims))

def sam_loss(y_pred, y_true):
    """Calcule la perte d'angle spectral (SAM) différentiable."""
    pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    true_flat = y_true.reshape(-1, y_true.shape[-1])
    
    dot_prod = torch.sum(pred_flat * true_flat, dim=1)
    norm_pred = torch.norm(pred_flat, p=2, dim=1)
    norm_true = torch.norm(true_flat, p=2, dim=1)
    
    cos_theta = dot_prod / (norm_pred * norm_true + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    
    sam_rad = torch.acos(cos_theta)
    return torch.mean(sam_rad)

def run_experiment(dataset_name="PaviaU", ranks=(16, 16, 8), epochs=1000, lr=2e-3,
                   lam_super=1.0, lam_sam=0.1, lam_S2=1.0, lam_S3=1.0, lam_sparse=1e-6):
    
    print("\n" + "="*80)
    print(f" EXPÉRIENCE : NL-JTAE v3 (ConvTranspose, Skips & SAM Loss) {dataset_name} | RANKS : {ranks} ")
    print("="*80)
    
    S = load_dataset(dataset_name, crop_size=(240, 240))
    H, W, C = S.shape
    
    (Y_S2_10m, Y_S2_20m, Y_S2_60m, Y_S3, 
     P_h_S3, P_w_S3, P_h_S2_20, P_w_S2_20, P_h_S2_60, P_w_S2_60,
     R_S3, R_S2_10, R_S2_20, R_S2_60) = simulate_observations(S)
    
    # Préparation des tenseurs
    Y_S2_10m_t = torch.tensor(Y_S2_10m, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    Y_S2_20m_t = torch.tensor(Y_S2_20m, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    Y_S2_20m_up = nn.functional.interpolate(Y_S2_20m_t, size=(H, W), mode='bilinear', align_corners=False).to(device)
    
    Y_S2_60m_t = torch.tensor(Y_S2_60m, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    Y_S2_60m_up = nn.functional.interpolate(Y_S2_60m_t, size=(H, W), mode='bilinear', align_corners=False).to(device)
    
    Y_S2_up = torch.cat([Y_S2_10m_t, Y_S2_20m_up, Y_S2_60m_up], dim=1) # (1, 13, 240, 240)
    
    Y_S3_t = torch.tensor(Y_S3, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    Y_S3_up = nn.functional.interpolate(Y_S3_t, size=(H, W), mode='bilinear', align_corners=False).to(device)
    
    # Concaténation conjointe (34 canaux)
    X_joint = torch.cat([Y_S2_up, Y_S3_up], dim=1) # (1, 34, 240, 240)
    
    # Cibles sur le GPU
    S_target = torch.tensor(S, dtype=torch.float32).unsqueeze(0).to(device)
    Y_S2_10m_target = Y_S2_10m_t
    Y_S2_20m_target = Y_S2_20m_t.permute(0, 2, 3, 1).to(device)
    Y_S2_60m_target = Y_S2_60m_t.permute(0, 2, 3, 1).to(device)
    Y_S3_target = Y_S3_t.permute(0, 2, 3, 1).to(device)
    
    # Création du modèle v3
    model = NonLinearJointTuckerAE(
        H, W, C, ranks,
        P_h_S3, P_w_S3, P_h_S2_20, P_w_S2_20, P_h_S2_60, P_w_S2_60,
        R_S3, R_S2_10, R_S2_20, R_S2_60
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = 200
    
    print("\nDébut de l'entraînement NL-JTAE v3...")
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        S_pred, y_S2_pred_full, y_S3_pred, G = model(X_joint)
        
        # 1. Perte Supervisée HSI (Reconstruction)
        loss_super = torch.mean((S_pred - S_target) ** 2)
        
        # 2. Perte SAM différentiable (Fidélité spectrale)
        loss_sam_term = sam_loss(S_pred, S_target)
        
        # 3. Pertes S2 à résolutions natives
        y_S2_pred_10m = y_S2_pred_full[:, :, :, :4]
        loss_S2_10m = torch.mean((y_S2_pred_10m.permute(0, 3, 1, 2) - Y_S2_10m_target) ** 2)
        
        y_S2_pred_20m_full = y_S2_pred_full[:, :, :, 4:10]
        y_S2_pred_20m_temp = torch.einsum('ij,bcjl->bcil', model.P_h_S2_20, y_S2_pred_20m_full.permute(0, 3, 1, 2))
        y_S2_pred_20m = torch.einsum('bcil,kl->bcik', y_S2_pred_20m_temp, model.P_w_S2_20).permute(0, 2, 3, 1)
        loss_S2_20m = torch.mean((y_S2_pred_20m - Y_S2_20m_target) ** 2)
        
        y_S2_pred_60m_full = y_S2_pred_full[:, :, :, 10:13]
        y_S2_pred_60m_temp = torch.einsum('ij,bcjl->bcil', model.P_h_S2_60, y_S2_pred_60m_full.permute(0, 3, 1, 2))
        y_S2_pred_60m = torch.einsum('bcil,kl->bcik', y_S2_pred_60m_temp, model.P_w_S2_60).permute(0, 2, 3, 1)
        loss_S2_60m = torch.mean((y_S2_pred_60m - Y_S2_60m_target) ** 2)
        loss_S2 = loss_S2_10m + loss_S2_20m + loss_S2_60m
        
        # 4. Perte S3 à résolution native
        loss_S3 = torch.mean((y_S3_pred - Y_S3_target) ** 2)
        
        # 5. Régularisation L1 sur le tenseur cœur
        loss_sparse = torch.mean(torch.abs(G))
        
        # Perte globale
        loss = (lam_super * loss_super + 
                lam_sam * loss_sam_term +
                lam_S2 * loss_S2 + 
                loss_S3 * lam_S3 + 
                lam_sparse * loss_sparse)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Évaluation périodique
        model.eval()
        with torch.no_grad():
            S_reconstructed = S_pred.squeeze(0).cpu().numpy()
            current_psnr = calculate_psnr(S, S_reconstructed)
            current_sam = calculate_sam(S, S_reconstructed)
            sparsity = float((torch.abs(G) < 1e-4).float().mean().item()) * 100
            
        scheduler.step(loss)
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.2e} (Super: {loss_super.item():.2e}, SAM_Loss: {loss_sam_term.item():.4f}) | PSNR: {current_psnr:.2f} dB | SAM: {current_sam:.2f}° | G Sparsity: {sparsity:.1f}%")
            
        if loss.item() < best_loss - 1e-7:
            best_loss = loss.item()
            epochs_no_improve = 0
            best_S_reconstructed = S_reconstructed.copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] Déclenché à l'époque {epoch}")
                break
                
    t_end = time.time()
    training_time = t_end - t_start
    print(f"\nEntraînement complété en {training_time:.2f} secondes.")
    
    # Métriques finales
    final_psnr = calculate_psnr(S, best_S_reconstructed)
    final_sam = calculate_sam(S, best_S_reconstructed)
    final_ergas = calculate_ergas(S, best_S_reconstructed, ratio=30)
    final_ssim = calculate_ssim(S, best_S_reconstructed)
    final_uiqi = calculate_uiqi(S, best_S_reconstructed)
    
    print("\n" + "="*50)
    print(" RÉSULTATS DE L'EVALUATION FINALE NL-JTAE v3 ")
    print("="*50)
    print(f"  - PSNR (Qualité Spatiale)        : {final_psnr:.2f} dB")
    print(f"  - SAM (Angle Spectral Moyen)     : {final_sam:.2f}°")
    print(f"  - ERGAS (Erreur globale adim.)   : {final_ergas:.4f}")
    print(f"  - SSIM (Similarité Structurelle) : {final_ssim:.4f}")
    print(f"  - UIQI (Qualité Universelle)     : {final_uiqi:.4f}")
    print("="*50)
    
    # Sauvegarde des visualisations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0d1117')
    rgb_bands = [60, 30, 2]
    
    axes[0, 0].imshow(S[..., rgb_bands])
    axes[0, 0].set_title("Ground Truth S (RGB)", color='white')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(best_S_reconstructed[..., rgb_bands])
    axes[0, 1].set_title(f"Fused NL-JTAE v3 (PSNR: {final_psnr:.1f}dB)", color='white')
    axes[0, 1].axis('off')
    
    diff = np.abs(S[..., rgb_bands] - best_S_reconstructed[..., rgb_bands])
    axes[0, 2].imshow(diff / (diff.max() + 1e-8))
    axes[0, 2].set_title("Absolute Difference Map", color='white')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(Y_S3_up.squeeze(0).permute(1, 2, 0).cpu().numpy()[..., [15, 10, 2]])
    axes[1, 0].set_title("S3 Input Upsampled (300m)", color='white')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(Y_S2_10m[..., [2, 1, 0]])
    axes[1, 1].set_title("S2 Input 10m bands (RGB)", color='white')
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.8, f"Modèle: NL-JTAE v3 (Transp+Skips)", color='white', fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.65, f"Rangs: {ranks}", color='#8b949e', fontsize=11)
    axes[1, 2].text(0.1, 0.5, f"PSNR: {final_psnr:.2f} dB", color='#58a6ff', fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.35, f"SAM: {final_sam:.2f}°", color='#3fb950', fontsize=11, fontweight='bold')
    axes[1, 2].text(0.1, 0.2, f"ERGAS: {final_ergas:.4f}", color='#ffa657', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plot_path = "./v3_nljtae_comparaison_paviau.png"
    plt.savefig(plot_path, dpi=150, facecolor='#0d1117')
    print(f"Figure sauvegardée sous : {plot_path}")
    
    # Rapatrier la figure
    import shutil
    shutil.copy(plot_path, "C:\\Users\\hamza\\.gemini\\antigravity\\brain\\15201f33-80da-4d2c-a3f4-2b3479314927\\v3_nljtae_comparaison_paviau.png")
    
    # Sauvegarde TIFF
    result_dest = "results/PaviaU_reconstructed_NL_JTAE_v3.tif"
    os.makedirs("results", exist_ok=True)
    tiff.imwrite(result_dest, best_S_reconstructed.astype(np.float32))
    print(f"Image HSI finale sauvegardée sous : {result_dest}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entraînement de NL-JTAE v3 avec ConvTranspose, Skips et perte SAM.")
    parser.add_argument('--dataset', type=str, default='PaviaU', help='Nom du jeu de données')
    parser.add_argument('--epochs', type=int, default=1000, help='Nombre d\'époques')
    parser.add_argument('--ranks', type=int, nargs=3, default=[16, 16, 32], help='Rangs de décomposition [R1, R2, R3]')
    parser.add_argument('--lr', type=float, default=2e-3, help='Taux d\'apprentissage')
    parser.add_argument('--lam_super', type=float, default=1.0, help='Poids perte reconstruction')
    parser.add_argument('--lam_sam', type=float, default=0.1, help='Poids perte SAM')
    parser.add_argument('--lam_S2', type=float, default=1.0, help='Poids perte S2')
    parser.add_argument('--lam_S3', type=float, default=1.0, help='Poids perte S3')
    parser.add_argument('--lam_sparse', type=float, default=1e-6, help='Poids régularisation L1')
    args = parser.parse_args()
    
    run_experiment(
        dataset_name=args.dataset,
        ranks=tuple(args.ranks),
        epochs=args.epochs,
        lr=args.lr,
        lam_super=args.lam_super,
        lam_sam=args.lam_sam,
        lam_S2=args.lam_S2,
        lam_S3=args.lam_S3,
        lam_sparse=args.lam_sparse
    )
