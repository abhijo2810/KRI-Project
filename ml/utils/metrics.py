import torch

def dice(pred, target, eps=1e-6):
    # pred/target are 0/1 tensors
    inter = (pred * target).sum()
    return (2*inter + eps) / (pred.sum() + target.sum() + eps)

def iou(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    union = (pred + target - pred*target).sum()
    return (inter + eps) / (union + eps)


def batch_dice(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    """
    Mean Dice over a batch of logit predictions.

    Args:
        logits:  (B, 1, H, W) raw model output
        targets: (B, 1, H, W) float mask {0, 1}
        thr:     threshold applied after sigmoid
    Returns:
        Mean Dice as a Python float
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > thr).float()
        scores = []
        for i in range(preds.shape[0]):
            scores.append(dice(preds[i], targets[i]).item())
    return float(sum(scores) / len(scores)) if scores else 0.0


def batch_iou(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    """Mean IoU over a batch of logit predictions."""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > thr).float()
        scores = []
        for i in range(preds.shape[0]):
            scores.append(iou(preds[i], targets[i]).item())
    return float(sum(scores) / len(scores)) if scores else 0.0
