"""
article/make_figures.py
─────────────────────────────────────────────────────────────────────────────
Figures de l'article (à lancer depuis la racine du projet) :
  fig3_visual_comparison.png : GT / v0 / v1 / v4 en composé RGB + cartes de
                               différence absolue,
  fig4_spectra.png           : spectres prédits vs référence en 4 pixels,
  fig5_sparsity.png          : histogramme du cœur G (si poids disponibles).
─────────────────────────────────────────────────────────────────────────────
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts_article.paper_common import load_paviau

RGB = [60, 30, 2]   # bandes pour le composé pseudo-RGB (convention du projet)

METHODS = [
    ("Tucker ALS (v0)", "results/paper_v0_reconstruction.tif"),
    ("CTAE (v1)", "results/paper_v1_reconstruction.tif"),
    ("MS-NL-JTAE (proposé)", "results/paper_v4_selfsup_reconstruction.tif"),
    ("MS-NL-JTAE (oracle)", "results/paper_v4_oracle_reconstruction.tif"),
]


def stretch(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def main():
    S = load_paviau()
    recs = [(name, tiff.imread(p).astype(np.float64)) for name, p in METHODS]

    # ── Fig. 3 : comparaison visuelle RGB + cartes d'erreur ──────────────────
    n = len(recs) + 1
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8.4))
    axes[0, 0].imshow(stretch(S[..., RGB]))
    axes[0, 0].set_title("Référence (GT)", fontsize=11)
    axes[1, 0].axis('off')
    for j, (name, R) in enumerate(recs, start=1):
        axes[0, j].imshow(stretch(R[..., RGB]))
        axes[0, j].set_title(name, fontsize=11)
        err = np.abs(S - R).mean(axis=-1)
        im = axes[1, j].imshow(err, cmap='inferno', vmin=0, vmax=np.percentile(err, 99))
        axes[1, j].set_title("|erreur| moyenne", fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("article/fig3_visual_comparison.png", dpi=150)
    plt.close()

    # ── Fig. 4 : spectres en 4 pixels tirés aléatoirement ─────────────────────
    rng = np.random.default_rng(0)
    pts = rng.integers(20, S.shape[0] - 20, size=(4, 2))
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6), sharey=True)
    for ax, (i, j) in zip(axes, pts):
        ax.plot(S[i, j], 'k-', lw=2, label='Référence')
        for name, R in recs:
            ax.plot(R[i, j], lw=1, label=name)
        ax.set_title(f"pixel ({i},{j})", fontsize=10)
        ax.set_xlabel("bande")
    axes[0].set_ylabel("réflectance")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("article/fig4_spectra.png", dpi=150)
    plt.close()

    print("Figures écrites : article/fig3_visual_comparison.png, article/fig4_spectra.png")


if __name__ == "__main__":
    main()
