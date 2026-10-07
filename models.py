import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import PRETRAINED


class ShapeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [1, 16, 32, 64, 96, 128]
        blocks = []
        for inputs, outputs in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(inputs, outputs, 4, 2, 1),
                    nn.GroupNorm(8, outputs),
                    nn.SiLU(),
                ]
            )
        self.encoder = nn.Sequential(*blocks)
        self.to_latent = nn.Linear(128 * 4 * 4, 64)
        self.from_latent = nn.Linear(64, 128 * 4 * 4)
        channels = [128, 96, 64, 32, 16, 16]
        blocks = []
        for inputs, outputs in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(inputs, outputs, 3, padding=1),
                    nn.GroupNorm(8, outputs),
                    nn.SiLU(),
                ]
            )
        self.decoder = nn.Sequential(*blocks, nn.Conv2d(16, 1, 3, padding=1))

    def encode(self, image):
        return self.to_latent(self.encoder(image).flatten(1))

    def forward(self, image):
        latent = self.encode(image)
        decoded = self.decoder(self.from_latent(latent).reshape(-1, 128, 4, 4))
        return decoded, latent


class PatternProjection(nn.Module):
    def __init__(self, mean=None, components=None):
        super().__init__()
        self.register_buffer(
            "mean",
            (
                torch.zeros(384)
                if mean is None
                else torch.as_tensor(mean, dtype=torch.float32)
            ),
        )
        self.linear = nn.Linear(384, 64, bias=False)
        if components is not None:
            self.linear.weight.data.copy_(torch.as_tensor(components))

    def forward(self, features):
        return F.normalize(self.linear(features - self.mean), dim=-1)


class TextHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, size),
                )
                for size in (16, 64, 64, 3)
            ]
        )

    def forward(self, features):
        palette, pattern, shape, weights = [head(features) for head in self.heads]
        palette = palette.reshape(-1, 4, 4)
        palette = torch.cat(
            [palette[:, :, :3].sigmoid(), palette[:, :, 3:].softmax(1)], dim=2
        )
        descriptor = torch.cat(
            [
                palette.flatten(1),
                F.normalize(pattern, dim=1),
                F.normalize(shape, dim=1),
            ],
            dim=1,
        )
        return descriptor, weights


def load_dino(device):
    model = torch.hub.load(
        str(PRETRAINED / "dinov2"), "dinov2_vits14", source="local", pretrained=False
    )
    model.load_state_dict(
        torch.load(
            PRETRAINED / "dinov2_small.pth", map_location="cpu", weights_only=True
        )
    )
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def dino_features(model, crops, batch_size=64):
    device = next(model.parameters()).device
    features = []
    for offset in range(0, len(crops), batch_size):
        images = (
            torch.as_tensor(
                crops[offset : offset + batch_size].copy(), device=device
            ).float()[:, None]
            / 255
        )
        images = (
            F.interpolate(
                images, (224, 224), mode="bicubic", align_corners=False, antialias=True
            )
            .clamp(0, 1)
            .repeat(1, 3, 1, 1)
        )
        mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        tokens = model.forward_features((images - mean) / std)["x_norm_patchtokens"]
        features.append(F.normalize(tokens.mean(1), dim=1).cpu().numpy())
    return np.concatenate(features)


def load_text_backbone(device):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

    folder = str(PRETRAINED / "fashion_clip")
    tokenizer = CLIPTokenizerFast.from_pretrained(folder, local_files_only=True)
    model = CLIPTextModelWithProjection.from_pretrained(folder, local_files_only=True)
    return tokenizer, model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def text_features(texts, tokenizer, model):
    device = next(model.parameters()).device
    tokens = tokenizer(
        texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)
    return F.normalize(model(**tokens).text_embeds.float(), dim=1)
