"""Post-training quantization (static, INT8) via torch.ao.quantization.

PTQ first because it's free if accuracy holds. If the macro-AUROC drop exceeds
2 percentage points we fall back to QAT (separate script — not implemented in
this module yet).
"""
from __future__ import annotations
from copy import deepcopy
import torch
import torch.nn as nn
import torch.ao.quantization as tq


class QuantWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant_f = tq.DeQuantStub()
        self.dequant_l = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        f, l = self.model(x)
        return self.dequant_f(f), self.dequant_l(l)


def static_ptq(model, calib_loader, device, n_calib_batches: int = 16):
    """Returns a quantized CPU model. Calibrates on a few val batches."""
    model = deepcopy(model).to("cpu").eval()
    qm = QuantWrapper(model)
    qm.qconfig = tq.get_default_qconfig("x86")
    tq.prepare(qm, inplace=True)
    with torch.no_grad():
        for i, (x_in, *_rest) in enumerate(calib_loader):
            qm(x_in.cpu())
            if i + 1 >= n_calib_batches:
                break
    tq.convert(qm, inplace=True)
    return qm
