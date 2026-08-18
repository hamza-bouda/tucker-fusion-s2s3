"""
run_nljtae_hyperbench.py
─────────────────────────────────────────────────────────────────────────────
Étude de faisabilité HyperBench du NL-JTAE sur PaviaU, protocole IDENTIQUE
au run historique `run_benchmark.py` (mode fast) : PSF gaussienne/Airy/sinc,
ratios ×4 et ×8, SNR 35/30 dB, métriques PSNR/SAM/ERGAS/SSIM.

Produit results_nljtae_fast.csv puis imprime le comparatif avec les CSV
existant de G-SCOTT-Tucker et du bicubique (mêmes cas, mêmes graines).
─────────────────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from hyperbench import BenchmarkConfig, DegradationSpec, run_benchmark
from modeles_non_lineaires.nljtae_hyperbench import NLJTAEAdapter

SCENE_PATH = 'pavia.mat'
SCENE_KEY = 'paviaU'

if __name__ == '__main__':
    MODE = sys.argv[1] if len(sys.argv) > 1 else 'fast'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    psf_names = ['gaussian', 'airy', 'sinc']
    specs = [
        DegradationSpec(4, 4, 35., 40.),
        DegradationSpec(8, 4, 30., 40.),
    ]

    cfg = BenchmarkConfig(
        scene_path=SCENE_PATH, scene_key=SCENE_KEY,
        psf_names=psf_names, psf_sigmas=[3.4], psf_kernel_radii=[7],
        degradation_specs=specs,
        metrics=['psnr', 'sam', 'ergas', 'ssim'],
        save_csv=True,
        output_csv_path=f'results_nljtae_{MODE}.csv',
        overwrite_csv_on_start=True, fail_fast=False,
    )

    print(f"=== NL-JTAE x HyperBench [{MODE}] — epochs={epochs} ===")
    run_benchmark(NLJTAEAdapter(epochs=epochs), cfg)

    # ── Comparatif avec les runs historiques (mêmes cas) ─────────────────────
    dn = pd.read_csv(f'results_nljtae_{MODE}.csv')
    frames = {'NL-JTAE (proposé)': dn}
    for label, path in [('G-SCOTT-Tucker', f'results_tucker_{MODE}.csv'),
                        ('Bicubic', f'results_bicubic_{MODE}.csv')]:
        try:
            frames[label] = pd.read_csv(path)
        except Exception:
            print(f"(CSV absent : {path})")

    print("\n" + "=" * 72)
    print("  COMPARATIF — moyennes sur tous les cas (PSF × ratios)")
    print("=" * 72)
    header = f"  {'Méthode':22s}" + "".join(f"{m:>10s}" for m in ['PSNR', 'SAM', 'ERGAS', 'SSIM'])
    print(header)
    print("  " + "-" * 66)
    for label, df in frames.items():
        if 'PSNR' not in df.columns:
            print(f"  {label:22s}  (aucun cas réussi)")
            continue
        vals = "".join(f"{df[m].mean():10.4f}" for m in ['PSNR', 'SAM', 'ERGAS', 'SSIM'])
        print(f"  {label:22s}{vals}")

    print("\n  Détail par ratio :")
    for r in sorted(dn['downsampling_ratio'].unique()):
        print(f"  -- ratio ×{r}")
        for label, df in frames.items():
            sub = df[df['downsampling_ratio'] == r]
            vals = "".join(f"{sub[m].mean():10.4f}" for m in ['PSNR', 'SAM', 'ERGAS', 'SSIM'])
            print(f"    {label:20s}{vals}")

    print("\n  Détail par PSF (PSNR / SAM) :")
    for psf in psf_names:
        line = f"  {psf:12s}"
        for label, df in frames.items():
            sub = df[df['psf_name'] == psf]
            line += f"  {label.split()[0]}: {sub['PSNR'].mean():.2f}/{sub['SAM'].mean():.2f}"
        print(line)
