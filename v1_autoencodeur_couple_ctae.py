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

# Détection de l'appareil de calcul
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de l'appareil : {device}")

# ── ÉTAPE 1 : CHARGEMENT DES JEUX DE DONNÉES ──────────────────────────────────

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = {
    "PaviaU": {
        "img_url": "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "img_file": "PaviaU.mat",
        "var_name": "paviaU"
    }
}

def download_dataset(name):
    cfg = DATASETS[name]
    dest_path = os.path.join(DATA_DIR, cfg["img_file"])
    if not os.path.exists(dest_path):
        print(f"Téléchargement du jeu de données {name}...")
        urllib.request.urlretrieve(cfg["img_url"], dest_path)
    return dest_path

def load_dataset(name, crop_size=(240, 240)):
    path = download_dataset(name)
    mat = sio.loadmat(path)
    cfg = DATASETS[name]
    img = mat[cfg["var_name"]].astype(np.float32)
    
    # Normalisation globale dans [0, 1]
    img_max = np.percentile(img, 99.9)
    img = np.clip(img / img_max, 0, 1)
    
    # Extraction du centre
    H, W, C = img.shape
    h_crop, w_crop = crop_size
    start_h = (H - h_crop) // 2
    start_w = (W - w_crop) // 2
    img = img[start_h:start_h+h_crop, start_w:start_w+w_crop, :]
    print(f"Image {name} chargée et recadrée à la taille {img.shape}")
    return img

# ── ÉTAPE 2 : SIMULATION SENTINEL-2 ET SENTINEL-3 (PROTOCOLE DE WALD MULTI-ÉCHELLE) ──

def get_gaussian_blur_matrix(dim, scale, sigma):
    """Génère la matrice de flou gaussien et de sous-échantillonnage 1D."""
    if scale == 1:
        return np.eye(dim)
    radius = int(4 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()
    
    H_blur = np.zeros((dim, dim))
    for i in range(dim):
        for j, val in enumerate(kernel):
            col = i + x[j]
            if 0 <= col < dim:
                H_blur[i, col] += val
    H_blur /= H_blur.sum(axis=1, keepdims=True)
    return H_blur[::scale, :]

def get_spectral_response_matrix(C_bands, c_bands):
    R = np.zeros((c_bands, C_bands))
    step = C_bands / c_bands
    for i in range(c_bands):
        start = int(i * step)
        end = int((i + 1) * step) if i < c_bands - 1 else C_bands
        R[i, start:end] = 1.0 / (end - start)
    return R

def simulate_observations(S):
    H, W, C = S.shape
    
    # 1. Simulation Sentinel-3 (scale = 30, 21 bandes)
    P_h_S3 = get_gaussian_blur_matrix(H, 30, 1.0)
    P_w_S3 = get_gaussian_blur_matrix(W, 30, 1.0)
    R_S3 = get_spectral_response_matrix(C, 21)
    
    Y_S3_temp = np.einsum('ij,jkl->ikl', P_h_S3, S)
    Y_S3_spatial = np.einsum('ijk,lj->ilk', Y_S3_temp, P_w_S3)
    Y_S3 = np.einsum('ijk,lk->ijl', Y_S3_spatial, R_S3)
    
    # 2. Simulation Sentinel-2 (13 bandes)
    # S2_10m (scale = 1, 4 bandes)
    R_S2_10m = get_spectral_response_matrix(C, 4)
    Y_S2_10m = np.einsum('ijk,lk->ijl', S, R_S2_10m)
    
    # S2_20m (scale = 2, 6 bandes)
    P_h_S2_20m = get_gaussian_blur_matrix(H, 2, 1.0)
    P_w_S2_20m = get_gaussian_blur_matrix(W, 2, 1.0)
    R_S2_20m = get_spectral_response_matrix(C, 6)
    Y_S2_20m_temp = np.einsum('ij,jkl->ikl', P_h_S2_20m, S)
    Y_S2_20m_spatial = np.einsum('ijk,lj->ilk', Y_S2_20m_temp, P_w_S2_20m)
    Y_S2_20m = np.einsum('ijk,lk->ijl', Y_S2_20m_spatial, R_S2_20m)
    
    # S2_60m (scale = 6, 3 bandes)
    P_h_S2_60m = get_gaussian_blur_matrix(H, 6, 1.0)
    P_w_S2_60m = get_gaussian_blur_matrix(W, 6, 1.0)
    R_S2_60m = get_spectral_response_matrix(C, 3)
    Y_S2_60m_temp = np.einsum('ij,jkl->ikl', P_h_S2_60m, S)
    Y_S2_60m_spatial = np.einsum('ijk,lj->ilk', Y_S2_60m_temp, P_w_S2_60m)
    Y_S2_60m = np.einsum('ijk,lk->ijl', Y_S2_60m_spatial, R_S2_60m)
    
    return (Y_S2_10m, Y_S2_20m, Y_S2_60m, Y_S3, 
            P_h_S3, P_w_S3, P_h_S2_20m, P_w_S2_20m, P_h_S2_60m, P_w_S2_60m,
            R_S3, R_S2_10m, R_S2_20m, R_S2_60m)

# ── ÉTAPE 3 : ARCHITECTURE DE L'AUTO-ENCODEUR TUCKER COUPLÉ ───────────────────

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
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.inception = InceptionBlock(64, 64)
        self.res_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(out_spatial_grid)
        
    def forward(self, x):
        out = self.init_conv(x)
        feat = self.inception(out)
        out = out + feat
        out = self.res_conv(out)
        G = self.pool(out)
        return G.permute(0, 2, 3, 1) # (B, R1, R2, R3)

class CoupledTuckerAE(nn.Module):
    def __init__(self, H, W, C_bands, ranks, 
                 P_h_S3, P_w_S3, P_h_S2_20, P_w_S2_20, P_h_S2_60, P_w_S2_60,
                 R_S3, R_S2_10, R_S2_20, R_S2_60,
                 init_factors=None):
        super().__init__()
        self.ranks = ranks
        R1, R2, R3 = ranks
        
        # Enregistrement des buffers physiques
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
        
        # Encodeurs couplés
        self.encoder_S2 = TuckerEncoder(in_channels=13, out_channels=R3, out_spatial_grid=(R1, R2), input_size=(H, W))
        self.encoder_S3 = TuckerEncoder(in_channels=21, out_channels=R3, out_spatial_grid=(R1, R2), input_size=(H, W))
        
        # Dictionnaires partagés
        init_A, init_B, init_C = (None, None, None)
        if init_factors is not None:
            init_A, init_B, init_C = init_factors
            
        if init_A is not None:
            self.A = nn.Parameter(torch.tensor(init_A, dtype=torch.float32))
        else:
            self.A = nn.Parameter(torch.randn(H, R1))
            
        if init_B is not None:
            self.B = nn.Parameter(torch.tensor(init_B, dtype=torch.float32))
        else:
            self.B = nn.Parameter(torch.randn(W, R2))
            
        if init_C is not None:
            self.C = nn.Parameter(torch.tensor(init_C, dtype=torch.float32))
        else:
            self.C = nn.Parameter(torch.randn(C_bands, R3))

    def forward(self, y_S2_up, y_S3_up):
        G_S2 = self.encoder_S2(y_S2_up)
        G_S3 = self.encoder_S3(y_S3_up)
        
        # 1. Super-résolution HSI complète
        S_pred = torch.einsum('brst,ir,js,kt->bijk', G_S2, self.A, self.B, self.C)
        
        # 2. Reconstruction Sentinel-2 complète (13 bandes)
        C_S2 = torch.matmul(self.R_S2, self.C)
        y_S2_pred_full = torch.einsum('brst,ir,js,kt->bijk', G_S2, self.A, self.B, C_S2)
        
        # 3. Reconstruction Sentinel-3 complète (21 bandes)
        A_S3 = torch.matmul(self.P_h_S3, self.A)
        B_S3 = torch.matmul(self.P_w_S3, self.B)
        C_S3 = torch.matmul(self.R_S3, self.C)
        y_S3_pred = torch.einsum('brst,ir,js,kt->bijk', G_S3, A_S3, B_S3, C_S3)
        
        return S_pred, y_S2_pred_full, y_S3_pred, G_S2, G_S3

# ── ÉTAPE 4 : INITIALISATION DES DICTIONNAIRES PAR SVD ─────────────────────────

def init_dictionaries_svd(S, ranks):
    R1, R2, R3 = ranks
    S_1 = S.transpose(0, 1, 2).reshape(S.shape[0], -1)
    U_A, _, _ = np.linalg.svd(S_1, full_matrices=False)
    A_init = U_A[:, :R1]
    
    S_2 = S.transpose(1, 0, 2).reshape(S.shape[1], -1)
    U_B, _, _ = np.linalg.svd(S_2, full_matrices=False)
    B_init = U_B[:, :R2]
    
    S_3 = S.transpose(2, 0, 1).reshape(S.shape[2], -1)
    U_C, _, _ = np.linalg.svd(S_3, full_matrices=False)
    C_init = U_C[:, :R3]
    
    return A_init, B_init, C_init

# ── ÉTAPE 5 : CALCUL DES MÉTRIQUES D'ÉVALUATION ────────────────────────────────

def calculate_psnr(ref, fused):
    rmse = np.sqrt(np.mean((ref - fused) ** 2))
    if rmse == 0: return 100.0
    return float(20 * np.log10(1.0 / rmse))

def calculate_sam(ref, fused):
    n_pixels = ref.shape[0] * ref.shape[1]
    ref_flat = ref.reshape(n_pixels, ref.shape[2])
    fused_flat = fused.reshape(n_pixels, fused.shape[2])
    
    dot_prod = np.sum(ref_flat * fused_flat, axis=1)
    norms = np.linalg.norm(ref_flat, axis=1) * np.linalg.norm(fused_flat, axis=1)
    
    cos_theta = np.clip(dot_prod / (norms + 1e-8), -1.0, 1.0)
    sam_rad = np.arccos(cos_theta)
    return float(np.mean(sam_rad) * 180.0 / np.pi)

def calculate_ergas(ref, fused, ratio=4):
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

def calculate_cc(ref, fused):
    n_bands = ref.shape[-1]
    ccs = []
    for b in range(n_bands):
        r = ref[..., b].flatten()
        f = fused[..., b].flatten()
        r_mean = r - np.mean(r)
        f_mean = f - np.mean(f)
        val = np.dot(r_mean, f_mean) / (np.linalg.norm(r_mean) * np.linalg.norm(f_mean) + 1e-8)
        ccs.append(val)
    return float(np.mean(ccs))

# ── ÉTAPE 6 : BOUCLE D'ENTRAÎNEMENT DE L'AUTO-ENCODEUR COUPLÉ ─────────────────

def run_experiment(dataset_name="PaviaU", ranks=(16, 16, 8), epochs=1000, lr=1e-3,
                   lam_super=1.0, lam_S2=1.0, lam_S3=1.0, lam_couple=1.0, lam_sparse=1e-6):
    
    print("\n" + "="*80)
    print(f" EXPÉRIENCE : CTAE {dataset_name} | RANKS : {ranks} ")
    print("="*80)
    
    # 1. Chargement de la Vérité Terrain
    S = load_dataset(dataset_name, crop_size=(240, 240))
    H, W, C = S.shape
    
    # 2. Simulation des Observations Sentinel-2 et Sentinel-3
    (Y_S2_10m, Y_S2_20m, Y_S2_60m, Y_S3, 
     P_h_S3, P_w_S3, P_h_S2_20, P_w_S2_20, P_h_S2_60, P_w_S2_60,
     R_S3, R_S2_10, R_S2_20, R_S2_60) = simulate_observations(S)
    
    # 3. Initialisation SVD des dictionnaires sur la vérité terrain
    print("\nInitialisation SVD des facteurs partagés...")
    A_init, B_init, C_init = init_dictionaries_svd(S, ranks)
    
    # 4. Upsampling initial pour l'entrée des encodeurs
    Y_S2_10m_t = torch.tensor(Y_S2_10m, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    Y_S2_20m_t = torch.tensor(Y_S2_20m, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    Y_S2_20m_up = nn.functional.interpolate(Y_S2_20m_t, size=(H, W), mode='bilinear', align_corners=False).to(device)
    
    Y_S2_60m_t = torch.tensor(Y_S2_60m, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    Y_S2_60m_up = nn.functional.interpolate(Y_S2_60m_t, size=(H, W), mode='bilinear', align_corners=False).to(device)
    
    Y_S2_up = torch.cat([Y_S2_10m_t, Y_S2_20m_up, Y_S2_60m_up], dim=1) # (1, 13, 240, 240)
    
    Y_S3_t = torch.tensor(Y_S3, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    Y_S3_up = nn.functional.interpolate(Y_S3_t, size=(H, W), mode='bilinear', align_corners=False).to(device)
    
    # Cibles d'entraînement sur le GPU
    S_target = torch.tensor(S, dtype=torch.float32).unsqueeze(0).to(device)
    Y_S2_10m_target = Y_S2_10m_t
    Y_S2_20m_target = Y_S2_20m_t.permute(0, 2, 3, 1).to(device)
    Y_S2_60m_target = Y_S2_60m_t.permute(0, 2, 3, 1).to(device)
    Y_S3_target = Y_S3_t.permute(0, 2, 3, 1).to(device)
    
    # 5. Création du modèle couplé
    model = CoupledTuckerAE(
        H, W, C, ranks,
        P_h_S3, P_w_S3, P_h_S2_20, P_w_S2_20, P_h_S2_60, P_w_S2_60,
        R_S3, R_S2_10, R_S2_20, R_S2_60,
        init_factors=(A_init, B_init, C_init)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = 200
    
    print("\nDébut de l'entraînement CTAE...")
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        S_pred, y_S2_pred_full, y_S3_pred, G_S2, G_S3 = model(Y_S2_up, Y_S3_up)
        
        # 1. Perte Supervisée HSI
        loss_super = torch.mean((S_pred - S_target) ** 2)
        
        # 2. Pertes S2 à résolutions natives
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
        
        # 3. Perte S3 à résolution native
        loss_S3 = torch.mean((y_S3_pred - Y_S3_target) ** 2)
        
        # 4. Perte de couplage latent
        loss_couple = torch.mean((G_S2 - G_S3) ** 2)
        
        # 5. Régularisation L1 sur le tenseur cœur
        loss_sparse = torch.mean(torch.abs(G_S2))
        
        # Perte globale
        loss = (lam_super * loss_super + 
                lam_S2 * loss_S2 + 
                lam_S3 * loss_S3 + 
                lam_couple * loss_couple + 
                lam_sparse * loss_sparse)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Évaluation
        model.eval()
        with torch.no_grad():
            S_reconstructed = S_pred.squeeze(0).cpu().numpy()
            current_psnr = calculate_psnr(S, S_reconstructed)
            current_sam = calculate_sam(S, S_reconstructed)
            sparsity = float((torch.abs(G_S2) < 1e-4).float().mean().item()) * 100
            
        scheduler.step(loss)
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.2e} (Super: {loss_super.item():.2e}, Couple: {loss_couple.item():.2e}) | PSNR: {current_psnr:.2f} dB | SAM: {current_sam:.2f}° | G Sparsity: {sparsity:.1f}%")
            
        # Early stopping
        if loss.item() < best_loss - 1e-7:
            best_loss = loss.item()
            epochs_no_improve = 0
            # Sauvegarder les meilleures reconstructions
            best_S_reconstructed = S_reconstructed.copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] Déclenché à l'époque {epoch}")
                break
                
    t_end = time.time()
    training_time = t_end - t_start
    print(f"\nEntraînement complété en {training_time:.2f} secondes.")
    
    # ── ÉTAPE 7 : RÉSULTATS ET SYNTHÈSE VISUELLE ────────────────────────────────
    
    final_psnr = calculate_psnr(S, best_S_reconstructed)
    final_sam = calculate_sam(S, best_S_reconstructed)
    final_ergas = calculate_ergas(S, best_S_reconstructed, ratio=4)
    final_cc = calculate_cc(S, best_S_reconstructed)
    
    print("\n" + "="*50)
    print(" RÉSULTATS DE L'EVALUATION FINALE CTAE ")
    print("="*50)
    print(f"  - PSNR (Qualité Spatiale)        : {final_psnr:.2f} dB")
    print(f"  - SAM (Angle Spectral Moyen)     : {final_sam:.2f}°")
    print(f"  - ERGAS (Erreur globale adim.)   : {final_ergas:.4f}")
    print(f"  - CC (Coefficient Corrélation)   : {final_cc:.4f}")
    print("="*50)
    
    # Génération des figures
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0d1117')
    
    # Bandes pour l'affichage RGB (ex: PaviaU bandes 60, 30, 2)
    rgb_bands = [60, 30, 2]
    
    axes[0, 0].imshow(S[..., rgb_bands])
    axes[0, 0].set_title("Ground Truth S (RGB)", color='white')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(best_S_reconstructed[..., rgb_bands])
    axes[0, 1].set_title(f"Fused CTAE (PSNR: {final_psnr:.1f}dB)", color='white')
    axes[0, 1].axis('off')
    
    # Différence absolue
    diff = np.abs(S[..., rgb_bands] - best_S_reconstructed[..., rgb_bands])
    axes[0, 2].imshow(diff / (diff.max() + 1e-8))
    axes[0, 2].set_title("Absolute Difference Map", color='white')
    axes[0, 2].axis('off')
    
    # Sentinel-3 observation upsampled
    axes[1, 0].imshow(Y_S3_up.squeeze(0).permute(1, 2, 0).cpu().numpy()[..., [15, 10, 2]])
    axes[1, 0].set_title("S3 Input Upsampled (300m)", color='white')
    axes[1, 0].axis('off')
    
    # Sentinel-2 observation (10m bands)
    axes[1, 1].imshow(Y_S2_10m[..., [2, 1, 0]])
    axes[1, 1].set_title("S2 Input 10m bands (RGB)", color='white')
    axes[1, 1].axis('off')
    
    # Légendes et métriques
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.8, f"Modèle: Tucker CTAE", color='white', fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.65, f"Rangs Tucker: {ranks}", color='#8b949e', fontsize=11)
    axes[1, 2].text(0.1, 0.5, f"PSNR: {final_psnr:.2f} dB", color='#58a6ff', fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.35, f"SAM: {final_sam:.2f}°", color='#3fb950', fontsize=11, fontweight='bold')
    axes[1, 2].text(0.1, 0.2, f"ERGAS: {final_ergas:.4f}", color='#ffa657', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plot_path = f"./comparaison_{dataset_name}_AE.png"
    plt.savefig(plot_path, dpi=150, facecolor='#0d1117')
    print(f"Figure sauvegardée sous : {plot_path}")
    
    # Sauvegarde TIFF
    result_dest = f"results/{dataset_name}_reconstructed_AE.tif"
    os.makedirs("results", exist_ok=True)
    tiff.imwrite(result_dest, best_S_reconstructed.astype(np.float32))
    print(f"Image HSI finale sauvegardée sous : {result_dest}")
    
    return final_psnr, final_sam, final_ergas, final_cc, plot_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-encodeur Tucker Couplé (CTAE) Sentinel-2 / Sentinel-3")
    parser.add_argument("--dataset", type=str, default="PaviaU")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--ranks", type=int, nargs=3, default=[16, 16, 8])
    parser.add_argument("--lam_super", type=float, default=1.0)
    parser.add_argument("--lam_S2", type=float, default=1.0)
    parser.add_argument("--lam_S3", type=float, default=1.0)
    parser.add_argument("--lam_couple", type=float, default=1.0)
    parser.add_argument("--lam_sparse", type=float, default=1e-6)
    args = parser.parse_args()
    
    run_experiment(
        dataset_name=args.dataset,
        ranks=tuple(args.ranks),
        epochs=args.epochs,
        lr=args.lr,
        lam_super=args.lam_super,
        lam_S2=args.lam_S2,
        lam_S3=args.lam_S3,
        lam_couple=args.lam_couple,
        lam_sparse=args.lam_sparse
    )
