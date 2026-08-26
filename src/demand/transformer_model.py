from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerDimensions:
    num_zones: int
    num_activities: int
    num_departure_bins: int
    num_duration_bins: int
    profile_dim: int


class ActivityTripTransformer(nn.Module):
    def __init__(
        self,
        dimensions: TransformerDimensions,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.max_sequence_length = max_sequence_length
        self.zone_embedding = nn.Embedding(dimensions.num_zones, d_model, padding_idx=0)
        self.activity_embedding = nn.Embedding(dimensions.num_activities, d_model, padding_idx=0)
        self.departure_embedding = nn.Embedding(dimensions.num_departure_bins, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_sequence_length, d_model)
        self.profile_projection = nn.Linear(dimensions.profile_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.normalization = nn.LayerNorm(d_model)
        self.destination_zone_head = nn.Linear(d_model, dimensions.num_zones)
        self.destination_activity_head = nn.Linear(d_model, dimensions.num_activities)
        self.duration_head = nn.Linear(d_model, dimensions.num_duration_bins)

    def forward(
        self,
        origin_zone_sequence: torch.Tensor,
        origin_activity_sequence: torch.Tensor,
        departure_bin_sequence: torch.Tensor,
        profile_features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, sequence_length = origin_zone_sequence.shape
        positions = torch.arange(sequence_length, device=origin_zone_sequence.device).unsqueeze(0)
        token = (
            self.zone_embedding(origin_zone_sequence)
            + self.activity_embedding(origin_activity_sequence)
            + self.departure_embedding(departure_bin_sequence)
            + self.position_embedding(positions)
            + self.profile_projection(profile_features).unsqueeze(1)
        )
        encoded = self.encoder(token, src_key_padding_mask=padding_mask)
        lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1
        final_state = encoded[torch.arange(batch_size, device=encoded.device), lengths]
        final_state = self.normalization(final_state)
        return {
            "destination_zone_logits": self.destination_zone_head(final_state),
            "destination_activity_logits": self.destination_activity_head(final_state),
            "duration_logits": self.duration_head(final_state),
        }
