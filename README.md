# Fusion hyperspectrale Sentinel-2 / Sentinel-3 par décomposition de Tucker couplée à cœur parcimonieux

Travail de stage Assistant Ingénieur — LISIC (ULCO), 2026.
**Auteur : Hamza BOUDA**

Super-résolution d'images satellitaires par fusion multi-capteurs : reconstruction
d'un cube à haute résolution spatiale ET spectrale à partir de Sentinel-2
(10/20/60 m, 13 bandes) et Sentinel-3/OLCI (300 m, 21 bandes).

## Approche

Le cœur du travail est une **décomposition de Tucker couplée dont le cœur G est
contraint à la parcimonie** (pénalité L1 / seuillage doux proximal), déclinée en :

- **Modèles linéaires** — même modèle, cinq solveurs comparés :
  ALS+FISTA, Adam+prox, Adam+L1, SGD+prox, LBFGS+prox
  (voir `rapport/rapport_modeles_lineaires.pdf` pour les architectures et les équations) ;
- **Modèles non-linéaires (NL-JTAE)** — auto-encodeur Tucker conjoint :
  v1 CTAE couplé, v3 (skips + ConvTranspose + perte SAM),
  v4 MS-NL-JTAE multi-entrées natif (pyramide alignée capteurs, strides 2×3×5) ;
- **Protocole auto-supervisé strict** — entraînement sur les seules observations
  via opérateurs PSF/SRF différentiables, jamais sur la référence.

## Organisation

| Dossier / fichiers | Contenu |
|---|---|
| `paper_common.py`, `paper_v0/v1/v4_*.py`, `paper_benchmark.py` | pipeline de benchmark 4 flux (ratio 30) |
| `linear_tucker_torch.py`, `run_linear_*.py` | modèles linéaires (5 solveurs) + évaluation HyperBench |
| `nljtae_hyperbench.py`, `run_nljtae_hyperbench.py` | NL-JTAE bi-flux pour le protocole HyperBench |
| `*_on_ritchie.py` | soumission des jobs GPU (OAR, cluster Ritchie) |
| `article/` | manuscrit IMRAD + figures |
| `rapport/` | rapport LaTeX des modèles linéaires |
| `results/` | métriques (CSV/JSON/MD) |

## Reproduire

1. Créer un environnement Python 3.11 avec `torch`, `numpy`, `scipy`,
   `matplotlib`, `pandas`, `scikit-image`, `paramiko`, `tensorly`.
2. Placer `PaviaU.mat` dans `data/` (non versionné).
3. Pour les scripts cluster : créer un fichier local `ritchie_secret.py`
   (non versionné) définissant `RITCHIE_PASSWORD`.
4. Benchmark local : `python paper_benchmark.py` — ou soumission GPU :
   `python paper_run_benchmark_on_ritchie.py`.
