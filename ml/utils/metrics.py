import torch

def dice(pred, target, eps=1e-6):
    # pred/target are 0/1 tensors
    inter = (pred * target).sum()
    return (2*inter + eps) / (pred.sum() + target.sum() + eps)

def iou(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    union = (pred + target - pred*target).sum()
    return (inter + eps) / (union + eps)
