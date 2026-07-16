# -*- coding: utf-8 -*-
"""
make_linear_comparison_figs.py
─────────────────────────────────────────────────────────────────────────────
Comparaison des algorithmes d'optimisation du modèle linéaire G-SCOTT-Tucker
sur le protocole HyperBench Pavia (6 cas identiques) :

    ALS + FISTA (historique) vs Adam+prox vs Adam+L1 vs SGD+prox vs LBFGS+prox
    + Étude d'influence des rangs
    + Courbe parcimonie-fidélité (Adam+prox vs Adam+L1)

Produit :
    article/fig5_linear_optimizers.png     métriques par méthode (barres)
    article/fig6_linear_convergence.png    courbes de convergence + temps
    article/fig7_linear_ranks.png           PSNR moyen vs Rangs Tucker
    article/fig8_sparsity_fidelity.png     Sparsité G et PSNR vs Beta
    results/linear_comparison_table.md     tableau récapitulatif avec meilleure config
─────────────────────────────────────────────────────────────────────────────
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = 'results'

# 1. Détection automatique de la meilleure configuration de tuning
tuning_tags = [
    "tuning_lr_3e-3_const",
    "tuning_lr_1e-2_const",
    "tuning_lr_3e-2_const",
    "tuning_lr_3e-3_cos",
    "tuning_lr_1e-2_cos",
    "tuning_lr_3e-2_cos",
]

best_tag = "tuning_lr_1e-2_const"  # fallback par défaut
best_psnr = -1.0
for t in tuning_tags:
    p = os.path.join(RES, f"results_linear_{t}_fast.csv")
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            avg_p = df['PSNR'].mean()
            if avg_p > best_psnr:
                best_psnr = avg_p
                best_tag = t
        except Exception as e:
            print(f"Erreur de lecture de {p} : {e}")

print(f"Meilleur tag de tuning détecté : {best_tag} avec {best_psnr:.2f} dB")

METHODS_TORCH = [best_tag, 'results_linear_adam_l1_fast', 'sgd_prox', 'lbfgs_prox']
# Pour les fichiers réels téléchargés :
# adam_prox est results_linear_adam_prox_fast.csv (historique ou nouveau)
# Nous allons mapper les clés proprement.

LABELS = {
    'als': 'ALS + FISTA (historique)',
    'adam_prox': 'Adam + prox (tuned)',
    'adam_l1': 'Adam + L1 (tuned)',
    'sgd_prox': 'SGD + prox',
    'lbfgs_prox': 'LBFGS + prox',
}
COLORS = {
    'als': '#555555',
    'adam_prox': '#1f77b4',
    'adam_l1': '#ff7f0e',
    'sgd_prox': '#2ca02c',
    'lbfgs_prox': '#9467bd'
}
METRICS = ['PSNR', 'SAM', 'ERGAS', 'SSIM']
SENSE = {'PSNR': 'haut', 'SAM': 'bas', 'ERGAS': 'bas', 'SSIM': 'haut'}

# Chargement des données
frames = {}
hists = {}

# ALS historique
als_path = os.path.join(RES, 'results_tucker_fast.csv')
if os.path.exists(als_path):
    frames['als'] = pd.read_csv(als_path)

# Adam prox tuned
p_tuned = os.path.join(RES, f'results_linear_{best_tag}_fast.csv')
if os.path.exists(p_tuned):
    frames['adam_prox'] = pd.read_csv(p_tuned)
    with open(os.path.join(RES, f'results_linear_{best_tag}_hist.json')) as f:
        hists['adam_prox'] = json.load(f)

# Adam L1 tuned (nous avons fait la recherche avec le même lr/scheduler que adam_prox best)
# Trouvons le tag correspondant pour L1 :
# Ex: si best_tag est tuning_lr_1e-2_cos, la version L1 est beta_sweep_l1_1e-6 (qui correspond à lr=1e-2 et scheduler=cosine)
# Pour être simple et générique, l1_tuned est lu de results_linear_beta_sweep_l1_1e-6_fast.csv
p_l1_tuned = os.path.join(RES, 'results_linear_beta_sweep_l1_1e-6_fast.csv')
if os.path.exists(p_l1_tuned):
    frames['adam_l1'] = pd.read_csv(p_l1_tuned)
    with open(os.path.join(RES, 'results_linear_beta_sweep_l1_1e-6_hist.json')) as f:
        hists['adam_l1'] = json.load(f)
else:
    # fallback sur l'historique non-tuned
    p_l1_hist = os.path.join(RES, 'results_linear_adam_l1_fast.csv')
    if os.path.exists(p_l1_hist):
        frames['adam_l1'] = pd.read_csv(p_l1_hist)
        with open(os.path.join(RES, 'results_linear_adam_l1_hist.json')) as f:
            hists['adam_l1'] = json.load(f)

# SGD
p_sgd = os.path.join(RES, 'results_linear_sgd_prox_fast.csv')
if os.path.exists(p_sgd):
    frames['sgd_prox'] = pd.read_csv(p_sgd)
    with open(os.path.join(RES, 'results_linear_sgd_prox_hist.json')) as f:
        hists['sgd_prox'] = json.load(f)

# LBFGS
p_lbfgs = os.path.join(RES, 'results_linear_lbfgs_prox_fast.csv')
if os.path.exists(p_lbfgs):
    frames['lbfgs_prox'] = pd.read_csv(p_lbfgs)
    with open(os.path.join(RES, 'results_linear_lbfgs_prox_hist.json')) as f:
        hists['lbfgs_prox'] = json.load(f)

keys = [k for k in ['als', 'adam_prox', 'adam_l1', 'sgd_prox', 'lbfgs_prox'] if k in frames]

# ── Figure 5 : métriques par méthode ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
x = np.arange(len(keys))
for ax, metric in zip(axes.ravel(), METRICS):
    vals = [frames[k][metric].mean() for k in keys]
    errs = [frames[k][metric].std() for k in keys]
    bars = ax.bar(x, vals, yerr=errs, capsize=4,
                  color=[COLORS[k] for k in keys], alpha=0.88)
    best = (np.argmax if SENSE[metric] == 'haut' else np.argmin)(vals)
    bars[best].set_edgecolor('crimson')
    bars[best].set_linewidth(2.2)
    for xi, v in zip(x, vals):
        ax.text(xi, v, f'{v:.2f}' if metric != 'SSIM' else f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k].replace(' ', '\n') for k in keys], fontsize=8)
    ax.set_title(f'{metric} ({SENSE[metric]} = mieux)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Modèle linéaire G-SCOTT-Tucker — même modèle, 5 optimiseurs\n'
             'HyperBench PaviaU : moyennes ± écart-type sur 6 cas (3 PSF × 2 ratios)',
             fontsize=12, fontweight='bold')
fig.tight_layout()
os.makedirs('article', exist_ok=True)
fig.savefig('article/fig5_linear_optimizers.png', dpi=150, bbox_inches='tight')
print('ok : article/fig5_linear_optimizers.png')

# ── Figure 6 : convergence (cas gaussian ×4) + temps ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# Légendes et courbes de convergence pour les méthodes PyTorch
methods_torch = [k for k in ['adam_prox', 'adam_l1', 'sgd_prox', 'lbfgs_prox'] if k in hists]

ax = axes[0]
for m in methods_torch:
    h = hists[m][0]                       # cas 0 = gaussian, ratio ×4
    ax.semilogy(h['iter'], h['loss'], color=COLORS[m], label=LABELS[m])
ax.set_xlabel('itération'); ax.set_ylabel('perte couplée (log)')
ax.set_title('Convergence — cas gaussian ×4')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
for m in methods_torch:
    losses = np.array([h['loss'] for h in hists[m]])   # (6 cas, n_pts)
    it = hists[m][0]['iter']
    med = np.median(losses, axis=0)
    ax.semilogy(it, med, color=COLORS[m], label=LABELS[m])
    ax.fill_between(it, losses.min(axis=0), losses.max(axis=0),
                    color=COLORS[m], alpha=0.15)
ax.set_xlabel('itération'); ax.set_ylabel('perte couplée (log)')
ax.set_title('Convergence — médiane et enveloppe des 6 cas')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
tcol = 'runtime_seconds'
times = [frames[k][tcol].mean() if tcol in frames[k].columns else np.nan for k in keys]
ax.bar(np.arange(len(keys)), times, color=[COLORS[k] for k in keys], alpha=0.88)
for xi, v in enumerate(times):
    if np.isfinite(v):
        ax.text(xi, v, f'{v:.1f}s', ha='center', va='bottom', fontsize=9)
ax.set_xticks(np.arange(len(keys)))
ax.set_xticklabels([LABELS[k].replace(' ', '\n') for k in keys], fontsize=8)
ax.set_title('Temps moyen par cas'); ax.grid(axis='y', alpha=0.3)

fig.suptitle('Dynamique d’optimisation du modèle linéaire G-SCOTT-Tucker (protocole identique)',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig('article/fig6_linear_convergence.png', dpi=150, bbox_inches='tight')
print('ok : article/fig6_linear_convergence.png')

# ── Figure 7 : PSNR vs rangs ──────────────────────────────────────────────────
rank_keys = ["(15,15,8)", "(20,20,10)", "(30,30,10)", "(40,40,15)"]
rank_files = [
    f"results_linear_{best_tag}_fast.csv",
    "results_linear_ranks_20_20_10_fast.csv",
    "results_linear_ranks_30_30_10_fast.csv",
    "results_linear_ranks_40_40_15_fast.csv"
]

rank_psnrs = []
rank_stds = []
valid_ranks = []

for label, filename in zip(rank_keys, rank_files):
    p = os.path.join(RES, filename)
    if os.path.exists(p):
        df = pd.read_csv(p)
        rank_psnrs.append(df['PSNR'].mean())
        rank_stds.append(df['PSNR'].std())
        valid_ranks.append(label)

if valid_ranks:
    fig, ax = plt.subplots(figsize=(6, 4))
    x_pos = np.arange(len(valid_ranks))
    ax.errorbar(x_pos, rank_psnrs, yerr=rank_stds, fmt='o-', color='#1f77b4',
                capsize=5, elinewidth=1.5, markeredgewidth=1.5, lw=2, label="G-SCOTT-Tucker")
    # Ajouter la baseline bicubique pour comparaison
    bicub_path = os.path.join(RES, 'results_bicubic_fast.csv')
    if os.path.exists(bicub_path):
        df_bic = pd.read_csv(bicub_path)
        bic_psnr = df_bic['PSNR'].mean()
        ax.axhline(bic_psnr, color='crimson', linestyle='--', label=f'Bicubique ({bic_psnr:.2f} dB)')
        
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_ranks)
    ax.set_xlabel('Rangs Tucker (R1, R2, R3)')
    ax.set_ylabel('PSNR moyen (dB)')
    ax.set_title('Influence des rangs Tucker sur la fidélité de reconstruction')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig('article/fig7_linear_ranks.png', dpi=150, bbox_inches='tight')
    print('ok : article/fig7_linear_ranks.png')

# ── Figure 8 : courbe sparsité-fidélité (Beta sweeps) ─────────────────────────
betas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
prox_sparsity = []
prox_psnr = []
l1_sparsity = []
l1_psnr = []

for b in betas:
    # Prox
    p_csv = os.path.join(RES, f"results_linear_beta_sweep_prox_{b}_fast.csv")
    p_json = os.path.join(RES, f"results_linear_beta_sweep_prox_{b}_hist.json")
    if os.path.exists(p_csv) and os.path.exists(p_json):
        df = pd.read_csv(p_csv)
        prox_psnr.append(df['PSNR'].mean())
        with open(p_json) as f:
            h = json.load(f)
            # moyenne de la dernière sparsité des 6 cas
            prox_sparsity.append(np.mean([case['sparsity'][-1] for case in h]))
            
    # L1
    l_csv = os.path.join(RES, f"results_linear_beta_sweep_l1_{b}_fast.csv")
    l_json = os.path.join(RES, f"results_linear_beta_sweep_l1_{b}_hist.json")
    if os.path.exists(l_csv) and os.path.exists(l_json):
        df = pd.read_csv(l_csv)
        l1_psnr.append(df['PSNR'].mean())
        with open(l_json) as f:
            h = json.load(f)
            l1_sparsity.append(np.mean([case['sparsity'][-1] for case in h]))

if len(prox_sparsity) == len(betas) and len(l1_sparsity) == len(betas):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # 1. Sparsité vs Beta
    ax1.semilogx(betas, prox_sparsity, 'o-', color='#1f77b4', lw=2, label='Prox (seuil doux exact)')
    ax1.semilogx(betas, l1_sparsity, 's--', color='#ff7f0e', lw=2, label='L1 (pénalité sous-gradient)')
    ax1.set_xlabel(r'Poids de la régularisation $\beta$ (log)')
    ax1.set_ylabel('Sparsité du cœur G (%)')
    ax1.set_title('Sparsité exacte du cœur G vs Beta')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # 2. PSNR vs Beta
    ax2.semilogx(betas, prox_psnr, 'o-', color='#1f77b4', lw=2, label='Prox')
    ax2.semilogx(betas, l1_psnr, 's--', color='#ff7f0e', lw=2, label='L1')
    ax2.set_xlabel(r'Poids de la régularisation $\beta$ (log)')
    ax2.set_ylabel('PSNR moyen (dB)')
    ax2.set_title('Fidélité de reconstruction (PSNR) vs Beta')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    fig.suptitle('Courbe Parcimonie-Fidélité : adam_prox vs adam_l1', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig('article/fig8_sparsity_fidelity.png', dpi=150, bbox_inches='tight')
    print('ok : article/fig8_sparsity_fidelity.png')

# ── Tableau récapitulatif ─────────────────────────────────────────────────────
lines = ['| Optimiseur | PSNR (dB) ↑ | SAM (°) ↓ | ERGAS ↓ | SSIM ↑ | t/cas (s) |',
         '|---|---|---|---|---|---|']
print()
print(f"{'Optimiseur':22s} {'PSNR':>8} {'SAM':>8} {'ERGAS':>8} {'SSIM':>8} {'t(s)':>7}")
for k in keys:
    df = frames[k]
    t = df[tcol].mean() if tcol in df.columns else float('nan')
    row = [df[m].mean() for m in METRICS]
    print(f"{LABELS[k]:22s} "
          f"{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f} {row[3]:8.3f} {t:7.1f}")
    lines.append(f"| {LABELS[k]} | {row[0]:.2f} | {row[1]:.2f} "
                 f"| {row[2]:.2f} | {row[3]:.3f} | {t:.1f} |")

# Ajouter les meilleures configurations de rangs si disponibles
best_rank_tag = None
best_rank_psnr = -1.0
for label, filename in zip(rank_keys[1:], rank_files[1:]): # à partir de (20,20,10)
    p = os.path.join(RES, filename)
    if os.path.exists(p):
        df = pd.read_csv(p)
        row = [df[m].mean() for m in METRICS]
        t = df[tcol].mean() if tcol in df.columns else float('nan')
        lines.append(f"| G-SCOTT {label} | {row[0]:.2f} | {row[1]:.2f} | {row[2]:.2f} | {row[3]:.3f} | {t:.1f} |")
        if row[0] > best_rank_psnr:
            best_rank_psnr = row[0]
            best_rank_tag = label

with open(os.path.join(RES, 'linear_comparison_table.md'), 'w', encoding='utf-8') as f:
    f.write('# Modèle linéaire G-SCOTT — comparaison des optimiseurs et des rangs (HyperBench PaviaU)\n\n'
            + '\n'.join(lines) + '\n')
print('\nok : results/linear_comparison_table.md')

