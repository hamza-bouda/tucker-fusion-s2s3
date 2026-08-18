# Chiffres clés — modèles linéaires Tucker

| Solveur | IP PSNR | IP SAM | IP Q2n | Pavia PSNR | Pavia SAM | Pavia Q2n | t/cas |
|---|---|---|---|---|---|---|---|
| ALS fermé | 10.17 | 22.80 | 0.063 | 12.84 | 20.45 | 0.045 | 240s |
| ALS Lipschitz
(v0) | 15.89 | 15.25 | 0.650 | 17.84 | 14.11 | 0.598 | 127s |
| Adam+prox | 19.00 | 11.22 | 0.790 | 19.53 | 10.28 | 0.635 | 94s |
| Adam+L1 | 19.00 | 11.22 | 0.790 | 19.53 | 10.27 | 0.635 | 87s |
| SGD+prox | 19.59 | 10.66 | 0.804 | 19.72 | 9.47 | 0.631 | 81s |
| LBFGS+prox | 19.08 | 10.97 | 0.816 | 18.67 | 12.10 | 0.592 | 64s |

## Sparsité obtenue (Pavia, prox)
- β=1e-06 : sparsité 0.0 %, PSNR 17.77 dB, SAM 11.73°
- β=1e-05 : sparsité 0.0 %, PSNR 17.77 dB, SAM 11.74°
- β=0.0001 : sparsité 0.0 %, PSNR 17.77 dB, SAM 11.72°
- β=0.001 : sparsité 0.0 %, PSNR 17.77 dB, SAM 11.73°
- β=0.01 : sparsité 0.8 %, PSNR 17.77 dB, SAM 11.73°
- β=0.1 : sparsité 45.6 %, PSNR 19.17 dB, SAM 10.02°
- β=1 : sparsité 100.0 %, PSNR 11.35 dB, SAM 19.20°
- β=10 : sparsité 100.0 %, PSNR 8.44 dB, SAM 26.31°
