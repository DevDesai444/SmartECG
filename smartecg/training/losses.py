"""Joint forecast + classification loss.

L = alpha * MSE(forecast, y_wave) + beta * BCE_with_logits(logits, y_lab)
"""
import torch
import torch.nn.functional as F


def joint_loss(forecast, logits, y_wave, y_lab, alpha=1.0, beta=1.0):
    mse = F.mse_loss(forecast, y_wave)
    bce = F.binary_cross_entropy_with_logits(logits, y_lab)
    return alpha * mse + beta * bce, {"mse": mse.detach(), "bce": bce.detach()}
