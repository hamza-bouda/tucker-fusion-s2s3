# gscott_tucker_hyperbench.py
import numpy as np
from hyperbench.methods.base import ReconstructionMethod
from hyperbench.core.types import ReconstructionInputs
from v0_lineaire_baseline_tucker_als import run_gscott_tucker

def build_psf_matrix(dim, ratio):
    """Construit une matrice de décimation simple pour HyperBench"""
    B = np.zeros((dim // ratio, dim))
    for i in range(dim // ratio):
        B[i, i*ratio:(i+1)*ratio] = 1.0 / ratio
    return B

class GSCOTTuckerAdapter(ReconstructionMethod):
    def __init__(self, ranks=(15, 15, 4), lam_H=1.0, lam_M=1.0, beta_factor=0.01, n_outer=15, n_fista=100):
        super().__init__(name='G-SCOTT', shape_policy='crop')
        self.ranks = ranks
        self.lam_H = lam_H
        self.lam_M = lam_M
        self.beta_factor = beta_factor
        self.n_outer = n_outer
        self.n_fista = n_fista

    def predict(self, inputs: ReconstructionInputs):
        # 1. Normalisation
        hr_msi = np.array(inputs.hr_msi)
        lr_hsi = np.array(inputs.lr_hsi)
        srf = np.array(inputs.srf)
        
        s_M = np.percentile(hr_msi, 99) + 1e-9
        M_norm = hr_msi / s_M
        H_norm = lr_hsi / s_M # On garde la même échelle pour la consistance
        
        # 2. Construction des matrices de dégradation
        H_sp, W_sp, c = M_norm.shape
        h_sp, w_sp, C = H_norm.shape
        r_sp = H_sp // h_sp
        
        Bh = build_psf_matrix(H_sp, r_sp)
        Bw = build_psf_matrix(W_sp, r_sp)
        
        sources = [
            (M_norm, np.eye(H_sp), np.eye(W_sp), srf, self.lam_M),
            (H_norm, Bh, Bw, np.eye(C), self.lam_H)
        ]
        
        # 3. Lancement de l'algorithme
        S_norm, G, _, _ = run_gscott_tucker(
            sources, self.ranks, self.beta_factor, self.n_outer, self.n_fista, verbose=False
        )
        
        # 4. Dénormalisation
        S_final = S_norm * s_M
        
        sp = float(np.mean(np.abs(G) < 1e-8))
        return S_final, {'sparsity': sp}