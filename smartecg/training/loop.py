"""Train / eval loops with AMP."""
from __future__ import annotations
import math
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import joint_loss
from .metrics import classification_metrics, forecast_metrics


def _amp_ctx(device, enabled):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def train_one_epoch(model, loader, optim, sched, device, cfg, scaler, step, wandb=None):
    model.train()
    log_every = cfg["train"]["log_every"]
    grad_clip = cfg["train"]["grad_clip"]
    alpha = cfg["train"]["loss_alpha"]
    beta = cfg["train"]["loss_beta"]
    amp = cfg["train"]["amp"] and device.type == "cuda"

    running = []
    for x_in, y_wave, y_lab, _meta in loader:
        x_in = x_in.to(device, non_blocking=True)
        y_wave = y_wave.to(device, non_blocking=True)
        y_lab = y_lab.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with _amp_ctx(device, amp):
            forecast, logits = model(x_in)
            loss, parts = joint_loss(forecast, logits, y_wave, y_lab, alpha, beta)

        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at step {step}: "
                  f"mse={parts['mse'].item():.3g} bce={parts['bce'].item():.3g}; skipping batch")
            continue

        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(gn):
                print(f"[warn] non-finite grad norm at step {step}: {gn}; zeroing grads")
                optim.zero_grad(set_to_none=True)
                continue
            optim.step()

        if sched is not None:
            sched.step()

        step += 1
        running.append(loss.item())
        if wandb is not None and step % log_every == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/mse": parts["mse"].item(),
                "train/bce": parts["bce"].item(),
                "train/lr": optim.param_groups[0]["lr"],
                "step": step,
            })
    return float(np.mean(running)) if running else math.nan, step


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    ys_lab, ss_lab, ys_wave, ps_wave = [], [], [], []
    for x_in, y_wave, y_lab, _meta in loader:
        x_in = x_in.to(device, non_blocking=True)
        y_wave = y_wave.to(device, non_blocking=True)
        forecast, logits = model(x_in)
        score = torch.sigmoid(logits)
        ys_lab.append(y_lab.numpy())
        ss_lab.append(score.cpu().numpy())
        ys_wave.append(y_wave.cpu().numpy())
        ps_wave.append(forecast.cpu().numpy())
    y_true_lab = np.concatenate(ys_lab, axis=0)
    y_score_lab = np.concatenate(ss_lab, axis=0)
    y_true_wave = np.concatenate(ys_wave, axis=0)
    y_pred_wave = np.concatenate(ps_wave, axis=0)
    cls = classification_metrics(y_true_lab, y_score_lab, classes)
    fct = forecast_metrics(y_true_wave, y_pred_wave)
    return cls, fct
