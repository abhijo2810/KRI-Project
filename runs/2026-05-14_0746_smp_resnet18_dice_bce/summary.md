# Run: 2026-05-14_0746_smp_resnet18_dice_bce

## Model Config
- Model:      smp_resnet18
- Loss:       dice_bce (α=0.3, β=0.7, γ=0.75)
- Epochs:     60
- LR:         0.001
- Batch size: 2
- ImageNet norm: True
- CLAHE:      False

## CV Results (5-fold stratified)

| Fold | Dice | IoU | Opt thr |
|------|------|-----|---------|
| 1 | 0.9202 | 0.8605 | 0.25 |
| 2 | 0.8900 | 0.8121 | 0.85 |
| 3 | 0.7963 | 0.7069 | 0.35 |
| 4 | 0.7222 | 0.6460 | 0.60 |
| 5 | 0.8181 | 0.7220 | 0.25 |

**Mean Dice = 0.8294 ± 0.0702**
**Mean IoU  = 0.7495 ± 0.0768**
Mean optimal threshold = 0.460

## Cover Correlation (n=24)
- Pearson r   = 0.877
- Spearman ρ  = 0.943
- Bland-Altman: bias = 0.745%  LoA [-40.25, 41.739]
