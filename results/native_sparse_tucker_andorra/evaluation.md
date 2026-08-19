# Native sparse Tucker fusion evaluation

Checkpoint: `/nfs/home/lisic/hbouda/tucker_fusion_native/outputs/native_sparse_tucker_andorra/best_checkpoint.pt`

## Native observation consistency

| Sensor | RMSE | PSNR (dB) | SAM (deg) | ERGAS | SSIM | UIQI |
|---|---:|---:|---:|---:|---:|---:|
| s2_10 | 0.370803 | 12.249 | 4.689 | 34.208 | 0.51715 | 0.41347 |
| s2_20 | 0.385999 | 11.882 | 2.061 | 34.915 | 0.49449 | 0.40548 |
| s2_60 | 0.289450 | 14.422 | 5.423 | 32.426 | 0.59057 | 0.67552 |
| olci | 0.270559 | 14.939 | 9.103 | 38.107 | 0.88641 | 0.65987 |

## Unsupervised diagnostics

- spectral_second_difference_l1: 0.16141907
- spatial_total_variation_l1: 0.00098621
- sparse_core_active_fraction_gt_1e-4: 0.85186632

The native metrics compare each decoded observation with its original sensor raster on the same grid; they are not a substitute for fused-cube ground truth.
