# Run: 2026-05-14_0746_smp_resnet18_tversky

## Model Config
- Model:      smp_resnet18
- Loss:       tversky (α=0.3, β=0.7, γ=0.75)
- Epochs:     60
- LR:         0.001
- Batch size: 2
- ImageNet norm: True
- CLAHE:      False

## CV Results (5-fold stratified)

| Fold | Dice | IoU | Opt thr |
|------|------|-----|---------|
| 1 | 0.9196 | 0.8611 | 0.10 |
| 2 | 0.9034 | 0.8330 | 0.90 |
| 3 | 0.7955 | 0.7202 | 0.90 |
| 4 | 0.3866 | 0.2977 | 0.30 |
| 5 | 0.7435 | 0.6432 | 0.90 |

**Mean Dice = 0.7497 ± 0.1931**
**Mean IoU  = 0.6710 ± 0.2024**
Mean optimal threshold = 0.620

## Cover Correlation (n=24)
- Pearson r   = 0.8483
- Spearman ρ  = 0.9164
- Bland-Altman: bias = 12.622%  LoA [-36.916, 62.161]
