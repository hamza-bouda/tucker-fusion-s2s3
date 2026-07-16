# math_utils.py
import numpy as np
import tensorly as tl
from tensorly import unfold

# Le backend sera défini par le script principal (ex: tl.set_backend('pytorch'))

def norm_cols(matrix):
    """
    Normalise les colonnes d'une matrice (Norme L2 = 1).
    Supporte Numpy et PyTorch (GPU) via Tensorly.
    """
    # tl.norm(matrix, axis=0) n'est pas toujours garanti selon le backend, on fait manuellement
    norms = tl.sqrt(tl.sum(matrix ** 2, axis=0))
    return matrix / (norms + 1e-12)

def soft_thresholding(X, threshold):
    """
    Opérateur proximal de la norme L1 (Seuillage doux) pour forcer la parcimonie.
    """
    return tl.sign(X) * tl.clip(tl.abs(X) - threshold, a_min=0, a_max=None)

def init_dictionary_svd(tensor, mode, rank):
    """
    Initialise un dictionnaire en extrayant les 'rank' premiers vecteurs 
    singuliers du tenseur déplié selon le mode donné, puis les normalise.
    """
    unfolded = unfold(tensor, mode)
    # Tensorly SVD (supporte GPU si backend PyTorch)
    U, _, _ = tl.tenalg.svd_interface(unfolded, n_eigenvecs=rank)
    return norm_cols(U[:, :rank])

import numpy as np
from scipy.ndimage import gaussian_filter

def estimer_matrices_degradation(S2_image, S3_image, facteur_echelle):
    """
    Estime physiquement la SRF (matrice R) et la PSF (flou) entre deux capteurs.
    
    S2_image : Tenseur (H, W, Bandes_S2) ou 4D (H, W, Bandes_S2, T)
    S3_image : Tenseur (h, w, Bandes_S3) ou 4D (H, W, Bandes_S3, T)
    """
    # Si c'est un tenseur 4D, on moyenne sur le temps pour la robustesse (ex: lisser les nuages)
    if S2_image.ndim == 4:
        S2_image = np.nanmean(S2_image, axis=3)
    if S3_image.ndim == 4:
        S3_image = np.nanmean(S3_image, axis=3)
        
    H, W, C_S2 = S2_image.shape
    h, w, C_S3 = S3_image.shape
    
    # ----------------------------------------------------
    # 1. LA PSF (Dégradation Spatiale)
    # Formule standard: FWHM = 2.355 * sigma ≈ facteur_echelle
    # ----------------------------------------------------
    sigma = facteur_echelle / 2.355
    print(f"  -> PSF : Flou Gaussien modélisé avec sigma = {sigma:.2f}")
    
    # On floute S2 pour simuler l'optique de S3
    S2_flou = np.zeros_like(S2_image)
    for c in range(C_S2):
        S2_flou[:, :, c] = gaussian_filter(S2_image[:, :, c], sigma=sigma)
        
    # On sous-échantillonne pour que S2 ait la même résolution que S3
    S2_degrade = S2_flou[::facteur_echelle, ::facteur_echelle, :]
    
    # Alignement précis des dimensions spatiales pour éviter les erreurs de mismatch
    h_min = min(S2_degrade.shape[0], S3_image.shape[0])
    w_min = min(S2_degrade.shape[1], S3_image.shape[1])
    
    S2_degrade = S2_degrade[:h_min, :w_min, :]
    S3_align = S3_image[:h_min, :w_min, :]
    
    # ----------------------------------------------------
    # 2. LA SRF (Dégradation Spectrale / Matrice R)
    # Résolution de l'équation S2_degrade = S3 @ R.T
    # ----------------------------------------------------
    # On aplatit les tenseurs en matrices 2D (Pixels, Bandes)
    pixels_s2 = S2_degrade.reshape(-1, C_S2)
    pixels_s3 = S3_align.reshape(-1, C_S3)
    
    # Calcul de la matrice pseudo-inverse
    # R.T = pinv(S3) * S2  => R = (pinv(S3) * S2).T
    print("  -> SRF : Calcul des moindres carrés en cours...")
    R_T = np.linalg.pinv(pixels_s3) @ pixels_s2
    R = R_T.T
    
    # Contraintes physiques de la matrice SRF
    # 1. La lumière ne peut pas être négative
    R = np.clip(R, 0, None) 
    # 2. Conservation de l'énergie (optionnel mais très recommandé)
    R = R / (np.sum(R, axis=1, keepdims=True) + 1e-8)
    
    print(f"  -> SRF : Matrice R estimée avec succès ! (Taille {R.shape})")
    
    return sigma, R

def create_spatial_degradation_matrix(dim, scale, sigma):
    """
    Crée une matrice 1D de dégradation spatiale combinant :
    1. Flou Gaussien (PSF)
    2. Sous-échantillonnage (Decimation)
    Cela empêche l'algorithme de créer des artefacts en grille (spikes).
    """
    radius = int(4 * sigma)
    if radius == 0:
        return np.eye(dim)[::scale, :]
        
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()
    
    H_blur = np.zeros((dim, dim))
    for i in range(dim):
        for j, val in enumerate(kernel):
            col = i + x[j]
            if 0 <= col < dim:
                H_blur[i, col] += val
                
    # Normalisation pour conserver l'énergie aux bords
    H_blur /= H_blur.sum(axis=1, keepdims=True)
    
    # Sous-échantillonnage
    return H_blur[::scale, :]