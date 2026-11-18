"""InrtlWrist release adapter. The masked reconstruction decoder is unused."""

import hashlib
import json

import torch
from torch import nn

from ..data import CLASSES, ROOT
from ..vendor.inrtlwrist import ViT1DEncoderAltAttn

RELEASE = ROOT / "models/InrtlWrist"
SHA256 = "38945472ad01a2285aab0a410f6abfaa34eced3334a3faa5c6dc3e4862d4dcfc"


def encoder(pretrained=True):
    config = json.loads((RELEASE / "config.json").read_text())
    net = ViT1DEncoderAltAttn(config)
    if pretrained:
        weights = RELEASE / config["weights"]
        with weights.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
                raise ValueError("InrtlWrist checkpoint digest differs from the release.")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        net.load_state_dict(state["model"], strict=True)
    return net


def classifier(seed):
    torch.manual_seed(seed)
    return nn.Linear(384, len(CLASSES))
