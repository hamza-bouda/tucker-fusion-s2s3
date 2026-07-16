# gscott_tucker_core.py
import numpy as np
import tensorly as tl
from tensorly import unfold
from tensorly.tenalg import multi_mode_dot
from math_utils import norm_cols, soft_thresholding, init_dictionary_svd

def update_Dn(sources, G, Ds, mode):
    Dn = Ds[mode]
    grad = tl.zeros_like(Dn)
    L_Dn = 0.0  
    
    n_modes = len(Ds)
    
    for X, Ps, lam in sources:
        # V_n : Projection de G sur les autres dictionnaires
        # identité du backend
        I_G = tl.eye(G.shape[mode], **tl.context(G))
        factors_V = [Ps[i] @ Ds[i] if i != mode else I_G for i in range(n_modes)]
        V_tensor = multi_mode_dot(G, factors_V, modes=list(range(n_modes)))
        V_n = unfold(V_tensor, mode) 
        
        factors_X = [Ps[i] @ Ds[i] for i in range(n_modes)]
        X_tilde = multi_mode_dot(G, factors_X, modes=list(range(n_modes)))
        Diff_n = unfold(X_tilde - X, mode)
        
        Pn = Ps[mode]
        grad += 2 * lam * (Pn.T @ Diff_n @ V_n.T)
        
        # norme de Frobenius si tensorly
        norm_Pn = tl.sqrt(tl.sum(Pn**2))
        norm_Vn = tl.sqrt(tl.sum(V_n**2))
        L_Dn += 2 * lam * (float(norm_Pn)**2) * (float(norm_Vn)**2)
        
    step = 1.0 / (L_Dn + 1e-8)
    Dn_new = Dn - step * grad
    return norm_cols(Dn_new)

def fista_G(sources, G_init, Ds, beta, n_iter=50):
    L_G = 0.0
    n_modes = len(Ds)
    for X, Ps, lam in sources:
        norm_prod = 1.0
        for i in range(n_modes):
            mat = Ps[i] @ Ds[i]
            norm_prod *= float(tl.sqrt(tl.sum(mat**2)))**2
        L_G += 2 * lam * norm_prod
        
    step = 1.0 / (L_G + 1e-8)
    Y = tl.copy(G_init)
    G_prev = tl.copy(G_init)
    t_prev = 1.0
    
    for i in range(n_iter):
        grad = tl.zeros_like(Y)
        for X, Ps, lam in sources:
            factors_X = [Ps[j] @ Ds[j] for j in range(n_modes)]
            X_tilde = multi_mode_dot(Y, factors_X, modes=list(range(n_modes)))
            Diff = X_tilde - X
            
            factors_diff = [(Ps[j] @ Ds[j]).T for j in range(n_modes)]
            grad += 2 * lam * multi_mode_dot(Diff, factors_diff, modes=list(range(n_modes)))
            
        G_new = soft_thresholding(Y - step * grad, beta * step)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_prev**2)) / 2.0
        Y = G_new + ((t_prev - 1.0) / t_new) * (G_new - G_prev)
        
        G_prev = G_new
        t_prev = t_new
        
    return G_prev

def total_loss(sources, G, Ds, beta):
    loss = 0.0
    n_modes = len(Ds)
    for X, Ps, lam in sources:
        factors_X = [Ps[i] @ Ds[i] for i in range(n_modes)]
        X_tilde = multi_mode_dot(G, factors_X, modes=list(range(n_modes)))
        loss += lam * float(tl.sum((X_tilde - X)**2))
    loss += beta * float(tl.sum(tl.abs(G)))
    return float(loss)

def run_gscott_tucker(sources, ranks, beta_factor=0.01, n_outer=15, n_fista=50, tol=1e-4, verbose=True):
    n_modes = len(ranks)
    M_main = sources[0][0]
    H_main = sources[-1][0]
    
    Ds = []
    for mode in range(n_modes):
        if mode < 2:
            Ds.append(init_dictionary_svd(M_main, mode, ranks[mode]))
        else:
            Ds.append(init_dictionary_svd(H_main, mode, ranks[mode]))
    
    Ps_main = sources[0][1]
    factors_init = [(Ps_main[i] @ Ds[i]).T for i in range(n_modes)]
    G = multi_mode_dot(M_main, factors_init, modes=list(range(n_modes)))

    L_G_init = 0.0
    for X, Ps, lam in sources:
        norm_prod = 1.0
        for i in range(n_modes):
            mat = Ps[i] @ Ds[i]
            norm_prod *= float(tl.sqrt(tl.sum(mat**2)))**2
        L_G_init += 2 * lam * norm_prod
        
    # Beta adaptatif : proportion de la norme de G (évite l'annulation totale)
    # On utilise une fraction de la norme initiale de G pour contrôler la sparsité
    norm_G_init = float(tl.sqrt(tl.sum(G**2)))
    beta = beta_factor * norm_G_init / max(np.prod(G.shape), 1)
    print(f"  Beta calculé = {beta:.4e} (norm_G={norm_G_init:.4e}, beta_factor={beta_factor})")

    prev_loss = None
    for it in range(n_outer):
        for mode in range(n_modes):
            Ds[mode] = update_Dn(sources, G, Ds, mode)
        
        G = fista_G(sources, G, Ds, beta, n_iter=n_fista)
        
        l_val = total_loss(sources, G, Ds, beta)
        if verbose:
            sp = float(tl.sum(tl.abs(G) < 1e-8)) / np.prod(G.shape) * 100
            print(f"Iter {it+1:02d}/{n_outer} | Loss: {l_val:.2e} | Sp: {sp:.1f}% | Norm G: {float(tl.sqrt(tl.sum(G**2))):.2e}")
            
        if prev_loss is not None and abs(prev_loss - l_val) / prev_loss < tol:
            if verbose:
                print(f"Convergence atteinte à l'itération {it+1} (tol={tol})")
            break
        prev_loss = l_val

    S_final = multi_mode_dot(G, Ds, modes=list(range(n_modes)))
    return S_final, G, Ds, beta