import os
import warnings

import torch


def select_device(requested=None):
    requested = requested or os.environ.get("FASHION_DEVICE", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            try:
                torch.ones(1, device="cuda").add_(1).cpu()
                return "cuda"
            except RuntimeError as error:
                warnings.warn(f"CUDA initialization failed; using CPU: {error}")
        return "cpu"
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Choose auto, cpu or cuda[:index]")
    return str(device)
