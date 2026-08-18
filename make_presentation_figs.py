# -*- coding: utf-8 -*-
"""
make_presentation_figs.py
─────────────────────────────────────────────────────────────────────────────
Figures pour la présentation hebdomadaire — modèles linéaires Tucker.

    presentation/figP1_metrics.png       5 solveurs × 2 scènes, 4 métriques
    presentation/figP2_convergence.png   dynamique d'optimisation
    presentation/figP3_sparsity.png      courbe parcimonie-fidélité (β)
    presentation/summary.md              chiffres clés pour les slides
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
OUT = 'presentation'
os.makedirs(OUT, exist_ok=True)

TAGS = ['als', 'als_lip', 'adam_prox', 'adam_l1', 'sgd_prox', 'lbfgs_prox']
LABELS = {'als': 'ALS fermé', 'als_lip': 'ALS Lipschitz\n(v0)',
          'adam_prox': 'Adam+prox', 'adam_l1': 'Adam+L1',
          'sgd_prox': 'SGD+prox', 'lbfgs_prox': 'LBFGS+prox'}
COLORS = {'als': '#555555', 'als_lip': '#8c8c8c', 'adam_prox': '#1f77b4',
          'adam_l1': '#ff7f0e', 'sgd_prox': '#2ca02c', 'lbfgs_prox': '#9467bd'}
SCENES = [('indianpines', 'Indian Pines'), ('pavia', 'Pavia University')]
METRICS = [('PSNR', 'dB', 'haut'), ('SAM', 'deg', 'bas'),
           ('q2n', '', 'haut'), ('ERGAS', '', 'bas')]

frames = {}
for s, _ in SCENES:
    for t in TAGS:
        p = os.path.join(RES, f'results_final_linear_{s}_{t}_fast.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            if 'PSNR' in df.columns:
                frames[(s, t)] = df

# ── Figure P1 : métriques 5 solveurs × 2 scènes ──────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
for row, (s, sname) in enumerate(SCENES):
    tags = [t for t in TAGS if (s, t) in frames]
    x = np.arange(len(tags))
    for col, (m, unit, sense) in enumerate(METRICS):
        ax = axes[row, col]
        vals = [frames[(s, t)][m].mean() for t in tags]
        errs = [frames[(s, t)][m].std() for t in tags]
        bars = ax.bar(x, vals, yerr=errs, capsize=3,
                      color=[COLORS[t] for t in tags], alpha=0.9)
        best = (np.argmax if sense == 'haut' else np.argmin)(vals)
        bars[best].set_edgecolor('crimson'); bars[best].set_linewidth(2.4)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[t] for t in tags], fontsize=7, rotation=20)
        title = {'q2n': 'Q2n'}.get(m, m)
        arrow = '↑' if sense == 'haut' else '↓'
        ax.set_title(f'{sname} — {title} {arrow}', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
fig.suptitle('Même modèle Tucker couplé sparse — cinq solveurs\n'
             'Moyennes sur 6 cas (PSF gaussienne/Airy/sinc × ratios 4/8, SNR 35/30 dB)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/figP1_metrics.png', dpi=150, bbox_inches='tight')
print('ok : figP1_metrics.png')

# ── Figure P2 : convergence (historique du sweep β=1e-6 prox, rangs 30) ──────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
hist_srcs = {'adam_prox': 'results_linear_beta_sweep_prox_1e-06_hist.json',
             'adam_l1': 'results_linear_beta_sweep_l1_1e-06_hist.json'}
ax = axes[0]
for t, fn in hist_srcs.items():
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        continue
    hists = json.load(open(p))
    losses = np.array([h['loss'] for h in hists])
    it = hists[0]['iter']
    ax.semilogy(it, np.median(losses, axis=0), color=COLORS[t], label=LABELS[t])
    ax.fill_between(it, losses.min(axis=0), losses.max(axis=0),
                    color=COLORS[t], alpha=0.15)
ax.set_xlabel('itération'); ax.set_ylabel('perte couplée (log)')
ax.set_title('Convergence — médiane et enveloppe des 6 cas (Pavia)')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
tcol = 'runtime_seconds'
tags = [t for t in TAGS if ('pavia', t) in frames]
times = [frames[('pavia', t)][tcol].mean() for t in tags]
ax.bar(np.arange(len(tags)), times, color=[COLORS[t] for t in tags], alpha=0.9)
for xi, v in enumerate(times):
    ax.text(xi, v, f'{v:.0f}s', ha='center', va='bottom', fontsize=9)
ax.set_xticks(np.arange(len(tags)))
ax.set_xticklabels([LABELS[t] for t in tags], fontsize=8)
ax.set_title('Temps moyen par cas (Pavia)'); ax.grid(axis='y', alpha=0.3)
fig.suptitle("Dynamique d'optimisation", fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/figP2_convergence.png', dpi=150, bbox_inches='tight')
print('ok : figP2_convergence.png')

# ── Figure P3 : courbe parcimonie-fidélité (balayage β) ──────────────────────
# deux conventions de nommage : beta_sweep_<fam>_<b> (petits β, run du 15/07)
# et beta_adam_<fam>_<b> (grands β, balayage étendu)
betas_files = {'1e-06': 1e-6, '1e-05': 1e-5, '0.0001': 1e-4,
               '0.001': 1e-3, '0.01': 1e-2, '1e-1': 1e-1, '1': 1.0, '10': 10.0}
data = {}
for fam, key in [('prox', 'Adam+prox (zéros exacts)'), ('l1', 'Adam+L1 (shrinkage)')]:
    bs, sp, ps, sam = [], [], [], []
    for btxt, bval in betas_files.items():
        cands = [os.path.join(RES, f'results_linear_beta_sweep_{fam}_{btxt}_fast.csv'),
                 os.path.join(RES, f'results_linear_beta_adam_{fam}_{btxt}_fast.csv')]
        p = next((q for q in cands if os.path.exists(q)), None)
        if p is None:
            continue
        df = pd.read_csv(p)
        if 'PSNR' not in df.columns:
            continue
        bs.append(bval)
        sp.append(df['sparsity_G'].mean())
        ps.append(df['PSNR'].mean())
        sam.append(df['SAM'].mean())
    data[key] = (bs, sp, ps, sam)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
styles = {'Adam+prox (zéros exacts)': ('#1f77b4', 'o-'),
          'Adam+L1 (shrinkage)': ('#ff7f0e', 's--')}
for key, (bs, sp, ps, sam) in data.items():
    col, st = styles[key]
    axes[0].semilogx(bs, sp, st, color=col, label=key)
    axes[1].semilogx(bs, ps, st, color=col, label=key)
    axes[2].semilogx(bs, sam, st, color=col, label=key)
axes[0].set_ylabel('sparsité exacte de G (%)')
axes[0].set_title('Parcimonie du cœur vs β')
axes[1].set_ylabel('PSNR (dB)')
axes[1].set_title('Fidélité vs β')
axes[2].set_ylabel('SAM (°)')
axes[2].set_title('Angle spectral vs β')
for ax in axes:
    ax.set_xlabel('β (poids L1)'); ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9)
fig.suptitle('Courbe parcimonie–fidélité : seuillage proximal vs pénalité L1 '
             '(Pavia, rangs 30×30×10)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/figP3_sparsity.png', dpi=150, bbox_inches='tight')
print('ok : figP3_sparsity.png')

# ── Chiffres clés pour les slides ─────────────────────────────────────────────
lines = ['# Chiffres clés — modèles linéaires Tucker\n']
lines.append('| Solveur | IP PSNR | IP SAM | IP Q2n | Pavia PSNR | Pavia SAM | Pavia Q2n | t/cas |')
lines.append('|---|---|---|---|---|---|---|---|')
for t in TAGS:
    row = [LABELS[t]]
    for s, _ in SCENES:
        df = frames.get((s, t))
        if df is None:
            row += ['—'] * 3
        else:
            row += [f"{df['PSNR'].mean():.2f}", f"{df['SAM'].mean():.2f}",
                    f"{df['q2n'].mean():.3f}" if 'q2n' in df.columns else '—']
    dfp = frames.get(('pavia', t))
    row.append(f"{dfp[tcol].mean():.0f}s" if dfp is not None else '—')
    lines.append('| ' + ' | '.join(row) + ' |')

lines.append('\n## Sparsité obtenue (Pavia, prox)')
bs, sp, ps, sam = data.get('Adam+prox (zéros exacts)', ([], [], [], []))
for b, s_, p_, sm in zip(bs, sp, ps, sam):
    lines.append(f'- β={b:g} : sparsité {s_:.1f} %, PSNR {p_:.2f} dB, SAM {sm:.2f}°')
open(f'{OUT}/summary.md', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('ok : summary.md')
print('\n'.join(lines[-8:]))
