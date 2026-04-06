from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class BertWordNetConfig:
    model_name: str = "bert-base-uncased"
    wordnet_feature_dim: int = 4
    dropout: float = 0.1
    num_labels: int = 2


class BertWordNetClassifier(nn.Module):
    """Binary classifier that fuses a BERT text representation with four normalized WordNet features."""

    def __init__(self, config: BertWordNetConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + config.wordnet_feature_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size // 2, config.num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        wordnet_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        fused_input = torch.cat([self.dropout(pooled_output), wordnet_features], dim=1)
        logits = self.classifier(fused_input)

        result = {"logits": logits}
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            result["loss"] = loss
        return result
