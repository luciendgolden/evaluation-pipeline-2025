from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import BaseModelOutput

if TYPE_CHECKING:
    import argparse


class ClassifierHead(nn.Module):

    def __init__(self, config: argparse.Namespace, hidden_size=None) -> None:
        super().__init__()
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        self.nonlinearity: nn.Sequential = nn.Sequential(
            nn.LayerNorm(hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size, config.num_labels)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.nonlinearity(embeddings)


class ModelForSequenceClassification(nn.Module):

    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__()
        self.transformer: nn.Module = AutoModel.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        model_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        hidden_size = model_config.hidden_size
        self.classifier: nn.Module = ClassifierHead(config, hidden_size)
        self.take_final = config.take_final

    def forward(self, input_data: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        output_transformer = self.transformer(input_data, attention_mask)
        if type(output_transformer) is tuple:
            encoding = output_transformer[0]
        elif type(output_transformer) is BaseModelOutput:
            if hasattr(output_transformer, "logits"):
                encoding = output_transformer.logits
            elif hasattr(output_transformer, "last_hidden_state"):
                encoding = output_transformer.last_hidden_state
            elif hasattr(output_transformer, "hidden_states"):
                encoding = output_transformer.hidden_states[-1]
            else:
                print("Unknown name for output of the model!")
                exit()
        else:
            print(f"Add support for output type: {type(output_transformer)}!")
            exit()
        if self.take_final:
            final_position = attention_mask[:, :, -1].squeeze().long().argmax(-1) - 1
            transformer_output = encoding[final_position].diagonal().t()
        else:
            transformer_output = encoding[:, 0]
        logits = self.classifier(transformer_output)

        return logits
