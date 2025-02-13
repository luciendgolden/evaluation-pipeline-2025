from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


class ClassifierHead(nn.Module):

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()
        self.nonlinearity: nn.Sequential = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(self, eembeddings: torch.Tensor) -> torch.Tensor:
        return self.nonlinearity(eembeddings)


def import_architecture(architecture):
    match architecture:
        case "base":
            from model import Bert
        case "extra":
            from model_extra import Bert
        case _:
            raise ValueError(f"The architecture cannot be {architecture}, it has to be one of the following: base, attglu, attgate, densemod, densesubmod, densecont, elc, qkln.")

    return Bert


class ModelForSequenceClassification(nn.Module):

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()
        self.transformer: nn.Module = import_architecture(config.architecture)(config)
        self.classifier: nn.Module = ClassifierHead(config)
        self.take_final = config.take_final

    def forward(self, input_data: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Make sure to change how the model takes in the inputs and attention_mask
        embedding = self.transformer.get_contextualized(input_data.t(), attention_mask.unsqueeze(1))
        if self.take_final:
            final_position = attention_mask[:, :, -1].squeeze().long().argmax(-1) - 1
            transformer_output = embedding[final_position].diagonal().t()
        else:
            transformer_output = embedding[0]
        logits = self.classifier(transformer_output)

        return logits
