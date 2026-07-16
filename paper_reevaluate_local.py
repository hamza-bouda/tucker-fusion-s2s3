"""
paper_reevaluate_local.py
─────────────────────────────────────────────────────────────────────────────
Ré-évaluation locale des reconstructions rapatriées de Ritchie : recalcule
le panel complet de métriques (dont SSIM, indisponible dans l'environnement
distant) contre la vérité terrain locale, et régénère le tableau final.
─────────────────────────────────────────────────────────────────────────────
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import tifffile as tiff

from paper_common import load_paviau, evaluate_all

RECONSTRUCTIONS = {
    "v0_tucker_als": "results/paper_v0_reconstruction.tif",
    "v1_ctae": "results/paper_v1_reconstruction.tif",
    "v4_msnljtae_selfsup": "results/paper_v4_selfsup_reconstruction.tif",
    "v4_msnljtae_oracle": "results/paper_v4_oracle_reconstruction.tif",
}

LABELS = {
    "v0_tucker_als": "Tucker ALS couplé (linéaire, v0)",
    "v1_ctae": "CTAE couplé (v1)",
    "v4_msnljtae_selfsup": "MS-NL-JTAE (proposé, auto-sup.)",
    "v4_msnljtae_oracle": "MS-NL-JTAE (oracle supervisé)",
}


def main():
    S = load_paviau()
    # Sparsité de G et temps : repris du JSON du run distant
    with open("results/paper_benchmark.json") as f:
        remote = json.load(f)

    results = {}
    for key, path in RECONSTRUCTIONS.items():
        S_hat = tiff.imread(path).astype(np.float64)
        m = evaluate_all(S, S_hat)
        m["Sparsity_G"] = remote.get(key, {}).get("Sparsity_G", float("nan"))
        m["time_s"] = remote.get(key, {}).get("time_s", float("nan"))
        results[key] = m

    with open("results/paper_final_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = ["| Méthode | PSNR (dB) | SAM (°) | ERGAS | SSIM | UIQI | Q2n | Sp(G) % | t (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, label in LABELS.items():
        m = results[key]
        lines.append(f"| {label} | {m['PSNR']:.2f} | {m['SAM']:.2f} | {m['ERGAS']:.4f} "
                     f"| {m['SSIM']:.4f} | {m['UIQI']:.4f} | {m['Q2n']:.4f} | {m['Sparsity_G']:.1f} "
                     f"| {m['time_s']:.0f} |")
    table = "\n".join(lines)
    with open("results/paper_final_table.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
