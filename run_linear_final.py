"""
run_linear_final.py
─────────────────────────────────────────────────────────────────────────────
Tableau FINAL de comparaison des modèles linéaires — toutes les métriques
(PSNR, SAM, Q2n, ERGAS + SSIM), sur DEUX jeux de données (format des tableaux
SCOTT/STEREO) :

    Pavia University (610×340×103)  +  Indian Pines (145×145×200)

Méthodes évaluées (même modèle Tucker couplé sparse, solveurs différents) :
    - ALS + FISTA          (solveur historique de la littérature)
    - Adam + prox (tuned)  (meilleurs lr/scheduler/rangs trouvés par le sweep)
    - Adam + L1  (tuned)
    - SGD + prox
    - LBFGS + prox

Les rangs optimaux sont lus automatiquement depuis les CSV du sweep de rangs
(results_linear_ranks_*_fast.csv) s'ils existent, sinon (20,20,10).

Produit : results_final_linear_<scene>_<tag>_fast.csv (un par méthode/scène)
          results_final_linear_table.md (tableau au format article)
─────────────────────────────────────────────────────────────────────────────
"""
import glob
import os
import re
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from hyperbench import BenchmarkConfig, DegradationSpec, run_benchmark
from run_linear_hyperbench import LinearTuckerAdapter, _calc_q2n, load_gt_scene

SCENES = [
    ('pavia', 'pavia.mat', 'paviaU'),
    ('indianpines', 'indian_pines.mat', 'indian_pines_corrected'),
]
SPECS = [DegradationSpec(4, 4, 35., 40.), DegradationSpec(8, 4, 30., 40.)]
PSFS = ['gaussian', 'airy', 'sinc']
BEST_LR, BEST_SCHED = 1e-2, 'cosine'   # issu du sweep lr/scheduler (17.77 dB)


def detect_best_ranks():
    """Choisit les rangs gagnants du sweep (fallback : 20,20,10)."""
    best, best_psnr = (20, 20, 10), -1.0
    for p in glob.glob('results_linear_ranks_*_fast.csv'):
        m = re.search(r'ranks_(\d+)_(\d+)_(\d+)_fast', p)
        if not m:
            continue
        try:
            psnr = pd.read_csv(p)['PSNR'].mean()
        except Exception:
            continue
        if psnr > best_psnr:
            best_psnr, best = psnr, tuple(map(int, m.groups()))
    print(f'Rangs retenus : {best} (PSNR sweep = {best_psnr:.2f} dB)')
    return best


class ALSAdapter(LinearTuckerAdapter):
    """ALS+FISTA historique, enrobé pour produire aussi le Q2n.
    Hérite de LinearTuckerAdapter uniquement pour le calcul GT/Q2n."""
    def __init__(self, ranks, scene_path, scene_key):
        super().__init__('adam_prox', ranks=ranks,
                         scene_path=scene_path, scene_key=scene_key)
        self.name = 'Tucker-lin (ALS+FISTA)'
        self._ranks_als = ranks

    def predict(self, inputs):
        # ALS réimplémenté ici avec les primitives du module historique :
        # la version actuelle de gstereo_tucker() y est cassée
        # (`D1, G = update_Dn(...)` alors que update_Dn ne renvoie que Dn).
        from gstereo_tucker_hyperbench import (update_Dn, fista_G, left_svd,
                                               norm_cols, build_psf_matrix)
        from tensorly import unfold
        from tensorly.tenalg import multi_mode_dot
        t0 = time.time()
        hr_msi = np.array(inputs.hr_msi)
        lr_hsi = np.array(inputs.lr_hsi)
        srf = np.array(inputs.srf)
        psf_kernel = np.array(inputs.psf)

        H_sp, W_sp, c = hr_msi.shape
        h_sp, w_sp, C = lr_hsi.shape
        r_sp = H_sp // h_sp
        s_M = np.percentile(hr_msi, 99) + 1e-9
        s_H = np.percentile(lr_hsi, 99) + 1e-9
        M1, H_n = hr_msi / s_M, lr_hsi / s_H

        BhH = build_psf_matrix(H_sp, r_sp, psf_kernel)
        BhW = build_psf_matrix(W_sp, r_sp, psf_kernel)
        IH, IW, I_C = np.eye(H_sp), np.eye(W_sp), np.eye(C)
        sources = [(H_n, BhH, BhW, I_C, 1.0), (M1, IH, IW, srf, 1.0)]

        r1, r2, r3 = self._ranks_als
        D1 = norm_cols(left_svd(unfold(M1, 0), r1))
        D2 = norm_cols(left_svd(unfold(M1, 1), r2))
        D3 = norm_cols(left_svd(unfold(H_n, 2), r3))
        G = multi_mode_dot(M1, [D1.T, D2.T, (srf @ D3).T], [0, 1, 2])
        G = G / (np.linalg.norm(G) + 1e-12)

        for _ in range(30):
            D1 = update_Dn(sources, G, D1, D2, D3, mode=0)
            D2 = update_Dn(sources, G, D1, D2, D3, mode=1)
            D3 = update_Dn(sources, G, D1, D2, D3, mode=2)
            G = G / (np.linalg.norm(G) + 1e-12)
            G = fista_G(sources, G, D1, D2, D3, 1e-6, 250)

        S = np.clip(multi_mode_dot(G, [D1, D2, D3], [0, 1, 2]) * s_H, 0., 1.)
        stats = {'fit_time_s': time.time() - t0,
                 'sparsity_G': float(np.mean(np.abs(G) < 1e-8)) * 100}
        self._add_q2n(S, stats)
        return S, stats

    def _add_q2n(self, S, stats):
        try:
            S_q = np.clip(S, 0.0, 1.0)
            H, W = S_q.shape[:2]
            gt = self._gt_scene[:H, :W, :S_q.shape[2]]
            stats['q2n'] = _calc_q2n(gt, S_q)
        except Exception:
            stats['q2n'] = float('nan')


if __name__ == '__main__':
    import sys
    only = (set(sys.argv[sys.argv.index('--only') + 1].split(','))
            if '--only' in sys.argv else None)
    ranks = detect_best_ranks()
    results = {}   # (scene, tag) -> DataFrame

    for scene_tag, scene_path, scene_key in SCENES:
        methods = [
            ('als', lambda: ALSAdapter(ranks, scene_path, scene_key)),
            ('adam_prox', lambda: LinearTuckerAdapter(
                'adam_prox', ranks=ranks, iters=10000, lr=BEST_LR,
                scheduler_type=BEST_SCHED, scene_path=scene_path, scene_key=scene_key)),
            ('adam_l1', lambda: LinearTuckerAdapter(
                'adam_l1', ranks=ranks, iters=10000, lr=BEST_LR,
                scheduler_type=BEST_SCHED, scene_path=scene_path, scene_key=scene_key)),
            ('sgd_prox', lambda: LinearTuckerAdapter(
                'sgd_prox', ranks=ranks, iters=10000, lr=BEST_LR,
                scheduler_type=BEST_SCHED, scene_path=scene_path, scene_key=scene_key)),
            ('lbfgs_prox', lambda: LinearTuckerAdapter(
                'lbfgs_prox', ranks=ranks, iters=300, lr=0.5,
                scene_path=scene_path, scene_key=scene_key)),
        ]
        for tag, make in methods:
            if only and tag not in only:
                continue
            csv = f'results_final_linear_{scene_tag}_{tag}_fast.csv'
            cfg = BenchmarkConfig(
                scene_path=scene_path, scene_key=scene_key,
                psf_names=PSFS, psf_sigmas=[3.4], psf_kernel_radii=[7],
                degradation_specs=SPECS,
                metrics=['psnr', 'sam', 'ergas', 'ssim'],
                save_csv=True, output_csv_path=csv,
                overwrite_csv_on_start=True, fail_fast=False,
            )
            print(f'\n=== [{scene_tag}] {tag} ===')
            try:
                run_benchmark(make(), cfg)
                df = pd.read_csv(csv)
                results[(scene_tag, tag)] = df
                if 'PSNR' in df.columns:
                    q = df['q2n'].mean() if 'q2n' in df.columns else float('nan')
                    print(f'  > PSNR {df.PSNR.mean():.2f} | SAM {df.SAM.mean():.2f} '
                          f'| Q2n {q:.3f} | ERGAS {df.ERGAS.mean():.2f}')
            except Exception as e:
                print(f'  ÉCHEC {scene_tag}/{tag} : {e}')

    # ── Tableau final au format article (Indian Pines | Pavia) ───────────────
    # Recharger depuis le disque tout ce qui existe (runs précédents inclus)
    for scene_tag, _, _ in SCENES:
        for tag in ('als', 'adam_prox', 'adam_l1', 'sgd_prox', 'lbfgs_prox'):
            p = f'results_final_linear_{scene_tag}_{tag}_fast.csv'
            if (scene_tag, tag) not in results and os.path.exists(p):
                try:
                    results[(scene_tag, tag)] = pd.read_csv(p)
                except Exception:
                    pass
    LABELS = {'als': 'ALS + FISTA', 'adam_prox': 'Adam + prox (tuned)',
              'adam_l1': 'Adam + L1 (tuned)', 'sgd_prox': 'SGD + prox',
              'lbfgs_prox': 'LBFGS + prox'}
    lines = ['| Méthode | ' + ' | '.join(
        f'{s} {m}' for s in ('IP', 'Pavia') for m in ('PSNR', 'SAM', 'Q2n', 'ERGAS')) + ' |',
        '|' + '---|' * 9]
    for tag, label in LABELS.items():
        row = [label]
        for scene_tag in ('indianpines', 'pavia'):
            df = results.get((scene_tag, tag))
            if df is None or 'PSNR' not in getattr(df, 'columns', []):
                row += ['—'] * 4
            else:
                q = df['q2n'].mean() if 'q2n' in df.columns else float('nan')
                row += [f'{df.PSNR.mean():.2f}', f'{df.SAM.mean():.2f}',
                        f'{q:.3f}', f'{df.ERGAS.mean():.2f}']
        lines.append('| ' + ' | '.join(row) + ' |')

    with open('results_final_linear_table.md', 'w') as f:
        f.write('# Modèles linéaires — métriques complètes '
                '(moyennes sur 6 cas : 3 PSF × ratios 4/8)\n\n' + '\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print('\nok : results_final_linear_table.md')
