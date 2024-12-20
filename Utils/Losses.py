import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)

    return 1 - loss.mean()


def combined_bce_dice_loss(pred, target, bce_weight=0.5, dice_weight=0.5):
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dice


def focal_loss(pred, target, alpha=0.25, gamma=2):
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


def combined_focal_dice_loss(pred, target, focal_weight=0.5, dice_weight=0.5):
    focal = focal_loss(pred, target)
    dice = dice_loss(pred, target)
    return focal_weight * focal + dice_weight * dice
