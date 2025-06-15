import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False  # leave perf on


def make_torch_generator(seed: int) -> torch.Generator:
    """Return a torch.Generator seeded for use with DataLoader(shuffle=True).
    Without this, DataLoader's shuffle is driven by the global RNG, which is
    consumed unpredictably by other code — so multi-seed runs end up with
    the same shuffle order and the seed-std collapses."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
