# Modèles linéaires — métriques complètes (moyennes sur 6 cas : 3 PSF × ratios 4/8)

| Méthode | IP PSNR | IP SAM | IP Q2n | IP ERGAS | Pavia PSNR | Pavia SAM | Pavia Q2n | Pavia ERGAS |
|---|---|---|---|---|---|---|---|---|
| ALS + FISTA | 10.17 | 22.80 | 0.063 | 16.03 | 12.84 | 20.45 | 0.045 | 19.62 |
| ALS Lipschitz (v0) | 15.89 | 15.25 | 0.650 | 8.32 | 17.84 | 14.11 | 0.598 | 10.11 |
| Adam + prox (tuned) | 19.00 | 11.22 | 0.790 | 5.66 | 19.53 | 10.28 | 0.635 | 8.96 |
| Adam + L1 (tuned) | 19.00 | 11.22 | 0.790 | 5.66 | 19.53 | 10.27 | 0.635 | 8.96 |
| SGD + prox | 19.59 | 10.66 | 0.804 | 5.36 | 19.72 | 9.47 | 0.631 | 8.88 |
| LBFGS + prox | 19.08 | 10.97 | 0.816 | 5.54 | 18.67 | 12.10 | 0.592 | 9.93 |
