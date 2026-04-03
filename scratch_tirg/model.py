"""Scratch PyTorch implementation of the TIRG model."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def tokenize(text: str) -> List[str]:
    """Tokenizes a modification string."""
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip().split()


@dataclass
class Vocabulary:
    """Simple vocabulary for modification text."""

    tokens: List[str]

    UNK = "<unk>"

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "Vocabulary":
        unique_tokens = sorted({tok for text in texts for tok in tokenize(text)})
        return cls([cls.UNK] + unique_tokens)

    @property
    def stoi(self) -> dict:
        return {token: idx for idx, token in enumerate(self.tokens)}

    @property
    def itos(self) -> List[str]:
        return self.tokens

    @property
    def unk_index(self) -> int:
        return 0

    def encode(self, text: str) -> List[int]:
        mapping = self.stoi
        encoded = [mapping.get(token, self.unk_index) for token in tokenize(text)]
        return encoded or [self.unk_index]


class TextEncoder(nn.Module):
    """Embedding + LSTM text encoder."""

    def __init__(self, vocab: Vocabulary, embed_dim: int):
        super().__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(len(vocab.tokens), embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim)
        self.fc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = [self.vocab.encode(text) for text in texts]
        lengths = [len(token_ids) for token_ids in encoded]
        batch = torch.zeros(
            (max(lengths), len(encoded)),
            dtype=torch.long,
            device=self.embedding.weight.device,
        )
        for batch_idx, token_ids in enumerate(encoded):
            batch[: len(token_ids), batch_idx] = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=batch.device,
            )
        embedded = self.embedding(batch)
        hidden = (
            torch.zeros(1, len(encoded), self.embed_dim, device=embedded.device),
            torch.zeros(1, len(encoded), self.embed_dim, device=embedded.device),
        )
        lstm_output, _ = self.lstm(embedded, hidden)
        text_features = [lstm_output[length - 1, idx, :] for idx, length in enumerate(lengths)]
        return self.fc_output(torch.stack(text_features, dim=0))


class ImageEncoder(nn.Module):
    """ResNet18 image encoder."""

    def __init__(self, embed_dim: int, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained and hasattr(torchvision.models, "ResNet18_Weights"):
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        elif pretrained:
            weights = True
        if hasattr(torchvision.models, "ResNet18_Weights"):
            backbone = torchvision.models.resnet18(weights=weights)
        else:
            backbone = torchvision.models.resnet18(pretrained=bool(weights))
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        backbone.fc = nn.Linear(backbone.fc.in_features, embed_dim)
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class TIRGComposer(nn.Module):
    """Gated + residual composition module from the TIRG paper."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.gating = nn.Sequential(
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.residual = nn.Sequential(
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        joined = torch.cat([image_features, text_features], dim=1)
        gate = self.gating(joined)
        residual = self.residual(joined)
        return torch.sigmoid(gate) * image_features * self.alpha[0] + residual * self.alpha[1]


class NormalizationLayer(nn.Module):
    """Learnable normalization layer used in the official implementation."""

    def __init__(self, normalize_scale: float = 4.0, learn_scale: bool = True):
        super().__init__()
        scale = torch.tensor(float(normalize_scale))
        if learn_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(features, dim=1, keepdim=True).clamp_min(1e-12)
        return self.scale * features / norms.expand_as(features)


def pairwise_squared_distance(
    x: torch.Tensor, y: torch.Tensor | None = None
) -> torch.Tensor:
    """Computes pairwise squared Euclidean distance."""
    if y is None:
        y = x
    x_norm = (x**2).sum(dim=1, keepdim=True)
    y_norm = (y**2).sum(dim=1, keepdim=True).transpose(0, 1)
    distances = x_norm + y_norm - 2.0 * x @ y.transpose(0, 1)
    return torch.clamp(distances, min=0.0)


def batch_soft_triplet_loss(
    composed_queries: torch.Tensor, target_images: torch.Tensor
) -> torch.Tensor:
    """Applies the same batch soft-triplet loss as the official implementation."""
    distances = pairwise_squared_distance(composed_queries, target_images)
    positives = distances.diag().unsqueeze(1)
    negatives = distances[~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)]
    negatives = negatives.view(distances.size(0), distances.size(1) - 1)
    return F.softplus(positives - negatives).mean()


class TIRGRetrievalModel(nn.Module):
    """End-to-end model for CSS3D retrieval."""

    def __init__(self, vocab: Vocabulary, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.vocab = vocab
        self.image_encoder = ImageEncoder(embed_dim=embed_dim, pretrained=pretrained)
        self.text_encoder = TextEncoder(vocab=vocab, embed_dim=embed_dim)
        self.composer = TIRGComposer(embed_dim=embed_dim)
        self.normalization = NormalizationLayer(normalize_scale=4.0, learn_scale=True)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_target_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)

    def compose_query(self, query_images: torch.Tensor, modifications: Sequence[str]) -> torch.Tensor:
        image_features = self.image_encoder(query_images)
        text_features = self.text_encoder(modifications)
        return self.composer(image_features, text_features)

    def training_loss(
        self,
        query_images: torch.Tensor,
        modifications: Sequence[str],
        target_images: torch.Tensor,
    ) -> torch.Tensor:
        composed_queries = self.normalization(self.compose_query(query_images, modifications))
        target_embeddings = self.normalization(self.encode_target_images(target_images))
        return batch_soft_triplet_loss(composed_queries, target_embeddings)
