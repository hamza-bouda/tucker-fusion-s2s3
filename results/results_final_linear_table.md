# Modèles linéaires — métriques complètes (moyennes sur 6 cas : 3 PSF × ratios 4/8)

| Méthode | IP PSNR | IP SAM | IP Q2n | IP ERGAS | Pavia PSNR | Pavia SAM | Pavia Q2n | Pavia ERGAS |
|---|---|---|---|---|---|---|---|---|
| ALS + FISTA | 10.17 | 22.80 | 0.063 | 16.03 | 12.84 | 20.45 | 0.045 | 19.62 |
| Adam + prox (tuned) | 18.51 | 11.88 | 0.769 | 5.95 | 19.07 | 10.32 | 0.578 | 9.46 |
| Adam + L1 (tuned) | 18.50 | 11.88 | 0.769 | 5.95 | 19.07 | 10.32 | 0.578 | 9.46 |
| SGD + prox | 18.35 | 12.12 | 0.728 | 6.04 | 18.74 | 10.57 | 0.520 | 10.00 |
| LBFGS + prox | 16.69 | 14.63 | 0.698 | 7.49 | 19.05 | 10.41 | 0.577 | 9.49 |
