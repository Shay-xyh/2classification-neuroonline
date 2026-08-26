"""CBraMod backbone and motor-imagery classification wrapper.

The implementation follows the official CBraMod release. Model-facing EEG is
expected in microvolts at 200 Hz and is split into non-overlapping one-second
patches before entering the foundation model.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PATCH_SIZE = 200
EMBEDDING_DIM = 200


class CrissCrossTransformerEncoderLayer(nn.Module):
    """Spatial-then-temporal attention block used by CBraMod."""

    def __init__(
        self,
        *,
        d_model: int = EMBEDDING_DIM,
        nhead: int = 8,
        dim_feedforward: int = 800,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_t = nn.MultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, source: Tensor) -> Tensor:
        output = source + self._attention_block(self.norm1(source))
        return output + self.dropout2(
            self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(output)))))
        )

    def _attention_block(self, inputs: Tensor) -> Tensor:
        batch, channels, patches, embedding = inputs.shape
        half = embedding // 2
        spatial = inputs[..., :half]
        temporal = inputs[..., half:]

        spatial = spatial.transpose(1, 2).reshape(batch * patches, channels, half)
        spatial = self.self_attn_s(
            spatial,
            spatial,
            spatial,
            need_weights=False,
        )[0]
        spatial = spatial.reshape(batch, patches, channels, half).transpose(1, 2)

        temporal = temporal.reshape(batch * channels, patches, half)
        temporal = self.self_attn_t(
            temporal,
            temporal,
            temporal,
            need_weights=False,
        )[0]
        temporal = temporal.reshape(batch, channels, patches, half)
        return self.dropout1(torch.cat((spatial, temporal), dim=-1))


class PatchEmbedding(nn.Module):
    def __init__(self, *, patch_size: int = PATCH_SIZE, d_model: int = EMBEDDING_DIM) -> None:
        super().__init__()
        if patch_size != PATCH_SIZE:
            raise ValueError(
                f"Official CBraMod weights require {PATCH_SIZE}-point patches, got {patch_size}."
            )
        self.patch_size = patch_size
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=(19, 7),
                padding=(9, 3),
                groups=d_model,
            ),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(patch_size), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(25, 25, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, patches, patch_size = inputs.shape
        if patch_size != self.patch_size:
            raise ValueError(
                f"CBraMod expected patch size {self.patch_size}, got {patch_size}."
            )
        flattened = inputs.reshape(batch, 1, channels * patches, patch_size)
        temporal = self.proj_in(flattened)
        temporal = temporal.permute(0, 2, 1, 3).reshape(
            batch,
            channels,
            patches,
            self.d_model,
        )
        spectrum = torch.abs(torch.fft.rfft(flattened, dim=-1, norm="forward"))
        spectral = self.spectral_proj(
            spectrum.reshape(batch, channels, patches, patch_size // 2 + 1)
        )
        embedded = temporal + spectral
        position = self.positional_encoding(embedded.permute(0, 3, 1, 2))
        return embedded + position.permute(0, 2, 3, 1)


class CBraModBackbone(nn.Module):
    """Official 12-layer CBraMod foundation backbone."""

    def __init__(self, *, n_layers: int = 12) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        layer = CrissCrossTransformerEncoderLayer()
        self.encoder = CrissCrossTransformerEncoder(layer, n_layers=n_layers)
        self.proj_out = nn.Sequential(nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM))

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.patch_embedding(inputs)
        return self.proj_out(self.encoder(output))


class CrissCrossTransformerEncoder(nn.Module):
    """Container named to remain checkpoint-compatible with the release."""

    def __init__(
        self,
        layer: CrissCrossTransformerEncoderLayer,
        *,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output


class CBraModClassifier(nn.Module):
    """CBraMod plus the official average-pooled downstream classifier."""

    def __init__(
        self,
        *,
        n_chans: int,
        n_times: int,
        n_classes: int,
        sfreq: float,
        pretrained_path: str | Path,
    ) -> None:
        super().__init__()
        if int(round(sfreq)) != PATCH_SIZE:
            raise ValueError(
                f"CBraMod requires model-facing EEG at {PATCH_SIZE} Hz, got {sfreq:g} Hz."
            )
        if n_times < PATCH_SIZE or n_times % PATCH_SIZE:
            raise ValueError(
                "CBraMod windows must contain a whole number of one-second "
                f"({PATCH_SIZE}-point) patches; got {n_times} points."
            )
        self.n_chans = int(n_chans)
        self.n_times = int(n_times)
        self.backbone = CBraModBackbone()
        weights_path = Path(pretrained_path).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(
                "CBraMod pretrained weights are missing: "
                f"{weights_path}. Run `python tools/download_cbramod_weights.py`."
            )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(EMBEDDING_DIM, n_classes),
        )

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"CBraMod expects [batch, channels, time], got {tuple(inputs.shape)}."
            )
        if inputs.shape[1:] != (self.n_chans, self.n_times):
            raise ValueError(
                "CBraMod input shape changed from "
                f"({self.n_chans}, {self.n_times}) to {tuple(inputs.shape[1:])}."
            )
        patches = inputs.reshape(
            inputs.shape[0],
            self.n_chans,
            self.n_times // PATCH_SIZE,
            PATCH_SIZE,
        )
        # The released MI downstream loader applies this exact /100 scaling.
        features = self.backbone(patches / 100.0)
        # NeuroOnline expects classifier input as [batch, embedding, tokens...].
        return features.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.forward_features(inputs))
