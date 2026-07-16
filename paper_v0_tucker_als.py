"""
paper_v0_tucker_als.py
─────────────────────────────────────────────────────────────────────────────
Baseline linéaire : décomposition de Tucker couplée parcimonieuse (type
STEREO/SCOTT avec cœur sparse, cf. CSTF) résolue par optimisation alternée :
  - dictionnaires D1 (H×R1), D2 (W×R2), D3 (C×R3) : descente de gradient
    avec pas de Lipschitz + normalisation des colonnes,
  - cœur G (R1×R2×R3) : FISTA avec seuillage doux (prox L1).

Modèle : Y_i ≈ G ×1 (P1_i D1) ×2 (P2_i D2) ×3 (R_i D3)  pour les 4 flux.
Régime strictement auto-supervisé : initialisation et pertes n'utilisent que
les observations et les opérateurs physiques.
─────────────────────────────────────────────────────────────────────────────
"""
import time
import numpy as np

from paper_common import load_paviau, simulate_streams, evaluate_all, format_metrics


# ── Algèbre tensorielle (numpy) ───────────────────────────────────────────────

def unfold(T, mode):
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def fold(M, mode, shape):
    full = [shape[mode]] + [s for i, s in enumerate(shape) if i != mode]
    return np.moveaxis(M.reshape(full), 0, mode)


def tucker_reconstruct(G, A, B, C):
    return np.einsum('rst,ir,js,kt->ijk', G, A, B, C, optimize=True)


def norm_cols(M):
    return M / (np.sqrt(np.sum(M ** 2, axis=0)) + 1e-12)


def soft_threshold(X, t):
    return np.sign(X) * np.maximum(np.abs(X) - t, 0.0)


# ── Mises à jour alternées ────────────────────────────────────────────────────

def update_dictionary(sources, G, Ds, mode):
    """Un pas de gradient (pas de Lipschitz) sur le dictionnaire du mode donné."""
    grad = np.zeros_like(Ds[mode])
    L = 0.0
    for Y, Ps, lam in sources:
        factors = [Ps[i] @ Ds[i] for i in range(3)]
        # V_n : G projeté sur les autres modes, déplié selon `mode`
        f_other = [factors[i] if i != mode else np.eye(G.shape[mode]) for i in range(3)]
        V = unfold(np.einsum('rst,ir,js,kt->ijk', G, *f_other, optimize=True), mode)
        diff = unfold(tucker_reconstruct(G, *factors) - Y, mode)
        grad += 2 * lam * (Ps[mode].T @ diff @ V.T)
        L += 2 * lam * np.sum(Ps[mode] ** 2) * np.sum(V ** 2)
    return norm_cols(Ds[mode] - grad / (L + 1e-8))


def fista_core(sources, G0, Ds, beta, n_iter=40):
    """FISTA sur le cœur G avec seuillage doux (opérateur proximal de L1)."""
    L = sum(2 * lam * np.prod([np.sum((Ps[i] @ Ds[i]) ** 2) for i in range(3)])
            for _, Ps, lam in sources)
    step = 1.0 / (L + 1e-8)
    Y_acc, G_prev, t_prev = G0.copy(), G0.copy(), 1.0
    for _ in range(n_iter):
        grad = np.zeros_like(Y_acc)
        for Yobs, Ps, lam in sources:
            factors = [Ps[i] @ Ds[i] for i in range(3)]
            diff = tucker_reconstruct(Y_acc, *factors) - Yobs
            grad += 2 * lam * np.einsum('ijk,ir,js,kt->rst', diff, *factors, optimize=True)
        G_new = soft_threshold(Y_acc - step * grad, beta * step)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_prev ** 2)) / 2.0
        Y_acc = G_new + ((t_prev - 1.0) / t_new) * (G_new - G_prev)
        G_prev, t_prev = G_new, t_new
    return G_prev


def total_loss(sources, G, Ds, beta):
    loss = sum(lam * np.sum((tucker_reconstruct(G, *[Ps[i] @ Ds[i] for i in range(3)]) - Y) ** 2)
               for Y, Ps, lam in sources)
    return float(loss + beta * np.sum(np.abs(G)))


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_tucker_als(S, obs, ops, ranks=(48, 48, 12), beta_factor=0.01,
                   n_outer=20, n_fista=40, tol=1e-4, verbose=True):
    H, W, C = S.shape
    R1, R2, R3 = ranks
    I_H, I_W = np.eye(H), np.eye(W)

    # (observation, (P1, P2, R), poids)
    sources = [
        (obs["Y10"], (I_H, I_W, ops["R_10"]), 1.0),
        (obs["Y20"], (ops["Ph_20"], ops["Pw_20"], ops["R_20"]), 1.0),
        (obs["Y60"], (ops["Ph_60"], ops["Pw_60"], ops["R_60"]), 1.0),
        (obs["Y300"], (ops["Ph_300"], ops["Pw_300"], ops["R_300"]), 2.0),
    ]

    # Initialisation depuis les observations uniquement :
    #  - modes spatiaux : SVD du flux S2-10m (pleine résolution),
    #  - mode spectral : SVD du flux S3 relevée par pseudo-inverse de la SRF.
    D1 = norm_cols(np.linalg.svd(unfold(obs["Y10"], 0), full_matrices=False)[0][:, :R1])
    D2 = norm_cols(np.linalg.svd(unfold(obs["Y10"], 1), full_matrices=False)[0][:, :R2])
    U3 = np.linalg.svd(unfold(obs["Y300"], 2), full_matrices=False)[0][:, :R3]
    D3 = norm_cols(np.linalg.pinv(ops["R_300"]) @ U3)
    Ds = [D1, D2, D3]

    # Cœur initial : projection moindres carrés de l'observation S3
    Ps3 = [ops["Ph_300"], ops["Pw_300"], ops["R_300"]]
    pinvs = [np.linalg.pinv(Ps3[i] @ Ds[i]) for i in range(3)]
    G = np.einsum('ijk,ri,sj,tk->rst', obs["Y300"], *pinvs, optimize=True)

    beta = beta_factor * np.linalg.norm(G) / max(G.size, 1)
    if verbose:
        print(f"  beta = {beta:.3e} | ranks = {ranks}")

    prev = None
    t0 = time.time()
    for it in range(n_outer):
        for mode in range(3):
            Ds[mode] = update_dictionary(sources, G, Ds, mode)
        G = fista_core(sources, G, Ds, beta, n_iter=n_fista)
        loss = total_loss(sources, G, Ds, beta)
        if verbose:
            sp = np.mean(np.abs(G) < 1e-8) * 100
            print(f"  iter {it+1:02d}/{n_outer} | loss {loss:.3e} | sparsité G {sp:.1f}%")
        if prev is not None and abs(prev - loss) / prev < tol:
            if verbose:
                print(f"  convergence (tol={tol}) à l'itération {it+1}")
            break
        prev = loss
    elapsed = time.time() - t0

    S_hat = np.clip(tucker_reconstruct(G, *Ds), 0, 1)
    return S_hat, G, elapsed


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Baseline Tucker ALS couplée parcimonieuse (v0)")
    p.add_argument('--ranks', type=int, nargs=3, default=[48, 48, 12])
    p.add_argument('--n_outer', type=int, default=20)
    p.add_argument('--beta_factor', type=float, default=0.01)
    args = p.parse_args()

    S = load_paviau()
    obs, ops = simulate_streams(S)
    S_hat, G, elapsed = run_tucker_als(S, obs, ops, ranks=tuple(args.ranks),
                                       beta_factor=args.beta_factor, n_outer=args.n_outer)
    m = evaluate_all(S, S_hat, G)
    m["time_s"] = elapsed
    print("\n" + format_metrics("v0 Tucker ALS (linéaire)", m))
    with open("results/paper_v0_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
