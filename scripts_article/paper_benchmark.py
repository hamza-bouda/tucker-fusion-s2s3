"""
paper_benchmark.py
─────────────────────────────────────────────────────────────────────────────
Benchmark complet de l'article sur PaviaU (protocole de Wald, ratio 30).

Toutes les méthodes sont exécutées dans le MÊME régime d'information :
seules les 4 observations natives et les opérateurs PSF/SRF sont accessibles
pendant l'estimation ; la vérité terrain ne sert qu'à l'évaluation finale.
Une ligne « oracle » supervisée du modèle proposé est ajoutée à titre de
borne supérieure, explicitement étiquetée.

Sortie : results/paper_benchmark.json + results/paper_benchmark_table.md
         + reconstructions TIF.
─────────────────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import json
import numpy as np
import tifffile as tiff

from scripts_article.paper_common import load_paviau, simulate_streams, evaluate_all, format_metrics
from scripts_article.paper_v0_tucker_als import run_tucker_als
from scripts_article.paper_v1_ctae import run_ctae
from scripts_article.paper_v4_msnljtae import run_msnljtae

os.makedirs("results", exist_ok=True)

def main(epochs=1500):
    S = load_paviau()
    obs, ops = simulate_streams(S)
    print(f"Scène : {S.shape} | flux natifs : "
          f"{obs['Y10'].shape}, {obs['Y20'].shape}, {obs['Y60'].shape}, {obs['Y300'].shape}\n")

    results = {}

    # ── v0 : baseline linéaire (Tucker ALS couplé parcimonieux) ──────────────
    print("=" * 70 + "\n v0 — Tucker ALS couplé parcimonieux (linéaire)\n" + "=" * 70)
    S0, G0, t0 = run_tucker_als(S, obs, ops, ranks=(48, 48, 12), n_outer=60)
    results["v0_tucker_als"] = {**evaluate_all(S, S0, G0), "time_s": t0}
    tiff.imwrite("results/paper_v0_reconstruction.tif", S0.astype(np.float32))
    print(format_metrics("v0 Tucker ALS", results["v0_tucker_als"]) + "\n")

    # ── v1 : CTAE couplé (analyse non-linéaire, synthèse multilinéaire) ──────
    print("=" * 70 + "\n v1 — CTAE couplé (auto-supervisé)\n" + "=" * 70)
    S1, G1, t1 = run_ctae(S, obs, ops, ranks=(24, 24, 12), epochs=epochs)
    results["v1_ctae"] = {**evaluate_all(S, S1, G1), "time_s": t1}
    tiff.imwrite("results/paper_v1_reconstruction.tif", S1.astype(np.float32))
    print(format_metrics("v1 CTAE", results["v1_ctae"]) + "\n")

    # ── v4 : MS-NL-JTAE multi-input natif (proposé, auto-supervisé) ──────────
    print("=" * 70 + "\n v4 — MS-NL-JTAE multi-input natif (auto-supervisé)\n" + "=" * 70)
    S4, G4, t4, hist4, model4 = run_msnljtae(S, obs, ops, r3=64, epochs=epochs)
    results["v4_msnljtae_selfsup"] = {**evaluate_all(S, S4, G4), "time_s": t4}
    tiff.imwrite("results/paper_v4_selfsup_reconstruction.tif", S4.astype(np.float32))
    print(format_metrics("v4 MS-NL-JTAE (selfsup)", results["v4_msnljtae_selfsup"]) + "\n")

    # ── v4 oracle : borne supérieure supervisée (étiquetée comme telle) ──────
    print("=" * 70 + "\n v4 — MS-NL-JTAE (oracle supervisé, borne supérieure)\n" + "=" * 70)
    S4o, G4o, t4o, _, _ = run_msnljtae(S, obs, ops, r3=64, epochs=epochs, supervised=True)
    results["v4_msnljtae_oracle"] = {**evaluate_all(S, S4o, G4o), "time_s": t4o}
    tiff.imwrite("results/paper_v4_oracle_reconstruction.tif", S4o.astype(np.float32))
    print(format_metrics("v4 MS-NL-JTAE (oracle)", results["v4_msnljtae_oracle"]) + "\n")

    # ── Synthèse ──────────────────────────────────────────────────────────────
    with open("results/paper_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = ["| Méthode | PSNR (dB) ↑ | SAM (°) ↓ | ERGAS ↓ | SSIM ↑ | UIQI ↑ | Sp(G) % | t (s) |",
             "|---|---|---|---|---|---|---|---|"]
    labels = {
        "v0_tucker_als": "Tucker ALS couplé (linéaire)",
        "v1_ctae": "CTAE couplé",
        "v4_msnljtae_selfsup": "MS-NL-JTAE (proposé, auto-sup.)",
        "v4_msnljtae_oracle": "MS-NL-JTAE (oracle supervisé)",
    }
    for k, label in labels.items():
        m = results[k]
        lines.append(f"| {label} | {m['PSNR']:.2f} | {m['SAM']:.2f} | {m['ERGAS']:.4f} "
                     f"| {m['SSIM']:.4f} | {m['UIQI']:.4f} | {m.get('Sparsity_G', float('nan')):.1f} "
                     f"| {m['time_s']:.0f} |")
    table = "\n".join(lines)
    with open("results/paper_benchmark_table.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)
    print("\nBenchmark terminé — résultats dans results/paper_benchmark.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Benchmark complet de l'article (PaviaU, Wald)")
    p.add_argument('--epochs', type=int, default=1500)
    args = p.parse_args()
    main(epochs=args.epochs)
