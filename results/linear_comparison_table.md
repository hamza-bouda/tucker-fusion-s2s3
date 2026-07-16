# Modèle linéaire — comparaison des optimiseurs (HyperBench PaviaU, 6 cas)

| Optimiseur | PSNR (dB) ↑ | SAM (°) ↓ | ERGAS ↓ | SSIM ↑ | t/cas (s) |
|---|---|---|---|---|---|
| ALS + FISTA (historique) | 13.68 | 20.07 | 18.19 | 0.124 | 265 |
| Adam + prox | 17.58 | 12.18 | 11.67 | 0.302 | 43 |
| Adam + L1 | 17.60 | 12.16 | 11.65 | 0.303 | 47 |
| SGD + prox | 16.65 | 13.94 | 13.25 | 0.223 | 46 |
