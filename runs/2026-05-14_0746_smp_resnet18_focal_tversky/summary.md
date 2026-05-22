# Run: 2026-05-14_0746_smp_resnet18_focal_tversky

## Model Config
- Model:      smp_resnet18
- Loss:       focal_tversky (α=0.3, β=0.7, γ=0.75)
- Epochs:     60
- LR:         0.001
- Batch size: 2
- ImageNet norm: True
- CLAHE:      False

## CV Results (5-fold stratified)

| Fold | Dice | IoU | Opt thr |
|------|------|-----|---------|
| 1 | 0.9207 | 0.8609 | 0.10 |
| 2 | 0.9022 | 0.8309 | 0.50 |
| 3 | 0.8578 | 0.7860 | 0.90 |
| 4 | 0.7578 | 0.6577 | 0.50 |
| 5 | 0.7384 | 0.6500 | 0.35 |

**Mean Dice = 0.8354 ± 0.0744**
**Mean IoU  = 0.7571 ± 0.0876**
Mean optimal threshold = 0.470

## Cover Correlation (n=24)
- Pearson r   = 0.9657
- Spearman ρ  = 0.986
- Bland-Altman: bias = 0.244%  LoA [-20.111, 20.6]
