from __future__ import annotations

import torch
from calculate_sentence_logits_scores import calculate_logits, calculate_scores_with_temperature

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@torch.no_grad()
def rank(sentences: list[str], model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, device: torch.device, batch_size: int, style: str, temperatures: torch.Tensor | None = None) -> list[list[int]]:
    """This function takes sentences, a model, and a tokenizer
    and outputs the ranking of the most likely sentence based
    on the model sentence logits.

    Args:
        sentences(list[str]): A list of sentences.
        model(torch.nn.Module): The model to evaluate.
        tokenizer(PreTrainedTokenizerBase): The tokenizer associated with the model.
        device(torch.device): The device the model is on.
        batch_size(int): The number of masked tokens to process together.
        style(str): The architecture style of the model (causal, mntp, mlm).
        temperatures(torch.Tensor | None): A tensor of
            temperatures to test the model at. (If None, the
            model will be evaluated at temperature 1)

    Returns:
        list[list[int]]: A ranking of the sentences evaluated
            based on the model sentence logits.
    """
    logits, sentences = calculate_logits(sentences, model, tokenizer, device, batch_size, style)
    scores = calculate_scores_with_temperature(logits, sentences, device, temperatures)

    ranking = torch.argsort(scores, dim=1, descending=True).tolist()

    return ranking
