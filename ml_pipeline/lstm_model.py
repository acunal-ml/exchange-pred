"""Attention-based LSTM (docs/03, section 2).

Stacked LSTM with an additive attention layer that weights critical
historical time steps before classification, plus dropout for
regularization. Returns attention weights alongside logits so a run can
log an interpretability artifact analogous to the LightGBM SHAP plot.
"""
from __future__ import annotations

import torch
from torch import nn


class AttentionLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_scores = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, n_features]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]

        scores = self.attn_scores(lstm_out).squeeze(-1)  # [batch, seq_len]
        attn_weights = torch.softmax(scores, dim=1)  # [batch, seq_len]
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden_size]

        context = self.dropout(context)
        logits = self.classifier(context)
        return logits, attn_weights
