import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss computed from logits. Operates over the full batch."""
    probs = torch.sigmoid(logits).view(-1)
    tgt   = targets.view(-1)
    inter = (probs * tgt).sum()
    return 1.0 - (2.0 * inter + eps) / (probs.sum() + tgt.sum() + eps)


def dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combined Dice + BCE loss.

    Args:
        logits:     (B, 1, H, W) raw model output (no sigmoid).
        targets:    (B, 1, H, W) float mask in {0.0, 1.0}.
        bce_weight: Weight for the BCE term; (1 - bce_weight) weights Dice.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * dice_loss(logits, targets)


def _tversky_index(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float,
) -> torch.Tensor:
    """
    Soft Tversky index computed from logits over the full batch.

    TI = (TP + ε) / (TP + α·FP + β·FN + ε)

    α weights false positives, β weights false negatives.
    Setting α=β=0.5 recovers the soft Dice index.
    Default α=0.3, β=0.7 penalises false negatives more heavily,
    boosting recall — useful when annotators under-paint colonies.
    """
    probs = torch.sigmoid(logits).view(-1)
    tgt   = targets.view(-1)
    tp    = (probs * tgt).sum()
    fp    = (probs * (1.0 - tgt)).sum()
    fn    = ((1.0 - probs) * tgt).sum()
    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Tversky loss: 1 - TI(α, β).

    α=0.3, β=0.7 → penalises false negatives more than false positives.
    """
    return 1.0 - _tversky_index(logits, targets, alpha, beta, eps)


def focal_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Focal Tversky loss: (1 - TI)^γ.

    γ < 1 applies a sub-linear scaling so that examples with very high
    Tversky loss (hard misses) contribute less, preventing the optimiser
    from over-focusing on the hardest — often noise-driven — pixels.
    Default γ=0.75 per Abraham & Khan (2019).
    """
    ti = _tversky_index(logits, targets, alpha, beta, eps)
    return (1.0 - ti) ** gamma
