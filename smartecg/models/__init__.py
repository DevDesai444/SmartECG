from .heads import ForecastHead, ClassifyHead
from .itransformer import iTransformer
from .lstm import LSTMEncoderModel
from .bilstm import BiLSTMEncoderModel
from .cnn1d import CNN1DEncoderModel
from .transformer_t import TransformerTEncoderModel


def build_model(cfg):
    name = cfg["model"]["name"]
    t_in = int(cfg["data"]["input_seconds"] * cfg["data"]["sampling_rate"])
    t_out = int(cfg["data"]["forecast_seconds"] * cfg["data"]["sampling_rate"])
    n_leads = cfg["num_leads"]
    n_classes = len(cfg["classes"])

    if name == "itransformer":
        return iTransformer(
            n_leads=n_leads, t_in=t_in, t_out=t_out, n_classes=n_classes,
            d_model=cfg["model"]["d_model"], n_heads=cfg["model"]["n_heads"],
            n_layers=cfg["model"]["n_layers"], d_ff=cfg["model"]["d_ff"],
            dropout=cfg["model"]["dropout"],
        )
    if name == "lstm":
        return LSTMEncoderModel(
            n_leads=n_leads, t_in=t_in, t_out=t_out, n_classes=n_classes,
            hidden=cfg["model"]["hidden"], n_layers=cfg["model"]["n_layers"],
            dropout=cfg["model"]["dropout"],
        )
    if name == "bilstm":
        return BiLSTMEncoderModel(
            n_leads=n_leads, t_in=t_in, t_out=t_out, n_classes=n_classes,
            hidden=cfg["model"]["hidden"], n_layers=cfg["model"]["n_layers"],
            dropout=cfg["model"]["dropout"],
        )
    if name == "cnn1d":
        return CNN1DEncoderModel(
            n_leads=n_leads, t_in=t_in, t_out=t_out, n_classes=n_classes,
            channels=cfg["model"]["channels"], kernel=cfg["model"]["kernel"],
            dropout=cfg["model"]["dropout"],
        )
    if name == "transformer_t":
        return TransformerTEncoderModel(
            n_leads=n_leads, t_in=t_in, t_out=t_out, n_classes=n_classes,
            patch=cfg["model"]["patch_size"], d_model=cfg["model"]["d_model"],
            n_heads=cfg["model"]["n_heads"], n_layers=cfg["model"]["n_layers"],
            d_ff=cfg["model"]["d_ff"], dropout=cfg["model"]["dropout"],
        )
    raise ValueError(f"unknown model: {name}")
