from __future__ import annotations

import torch
import torch.nn.functional as F

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _create_attention_mask(input_ids: torch.Tensor, padding: int, style: str = "bidirectional", device: torch.device = torch.device("cpu")) -> torch.Tensor:
    attention_mask = None

    if style == "bidirectional":
        attention_mask = torch.ones_like(input_ids, dtype=torch.int, device=device)
        attention_mask[:, attention_mask.size(-1) - padding:] = 0

    return attention_mask


@torch.no_grad()
def calculate_logits(sentences: list[str], model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, device: torch.device, batch_size: int, style: str) -> tuple[torch.Tensor, list[list[int]]]:
    """This function takes a pairs of sentences, a model, and a tokenizer and outputs the
    tokenized sentences as well as their logits given by the model for a specified
    architectural style.

    Args:
        sentences(list[str]): A list of sentences.
        model(torch.nn.Module): The model to evaluate.
        tokenizer(PreTrainedTokenizerBase): The tokenizer associated with the model.
        device(torch.device): The device the model is on.
        batch_size(int): The number of masked tokens to process together.
        style(str): The architecture style of the model (causal, mntp, mlm).

    Returns:
        torch.Tensor: The logits of each sentence.
        list[list[int]]: The tokenized sentences.
    """
    mask_index = tokenizer.mask_token_id
    pad_index = tokenizer.pad_token_id
    cls_index = torch.tensor([tokenizer.cls_token_id])
    sep_index = torch.tensor([tokenizer.sep_token_id])

    sentences = [torch.tensor(tokenizer(s, add_special_tokens=False).input_ids) for s in sentences]

    def _prepare_mlm(tokens, padding: int, device: str | torch.device = "cpu"):
        tokens = torch.cat([cls_index, tokens, sep_index, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - 2 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[1:-(1 + padding), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = _create_attention_mask(input_ids, padding, device=device)
        return input_ids, attention_mask

    def _prepare_mntp(tokens, padding: int, device: str | torch.device = "cpu"):
        tokens = torch.cat([cls_index, tokens, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - 1 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[1:(-padding if padding > 0 else None), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = _create_attention_mask(input_ids, padding, device=device)
        return input_ids, attention_mask

    def _prepare_causal(tokens, padding: int, device: str | torch.device = "cpu"):
        input_ids = torch.cat([cls_index, tokens, torch.full((padding,), fill_value=pad_index)]).to(device)
        return input_ids

    max_length = max(s.size(0) for s in sentences)
    if style == "mlm":
        input_ids, attention_masks = zip(*[_prepare_mlm(s, max_length - s.size(0), device) for s in sentences])
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0)
        indices = [torch.arange(1, 1 + len(s), device=device) for s in sentences]
    elif style == "causal":
        input_ids = [_prepare_causal(s, max_length - s.size(0), device) for s in sentences]
        input_ids = torch.stack(input_ids, dim=0)
        indices = [torch.arange(0, len(s), device=device) for s in sentences]
    elif style == "mntp":
        input_ids, attention_masks = zip(*[_prepare_mntp(s, max_length - s.size(0), device) for s in sentences])
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0)
        indices = [torch.arange(0, len(s), device=device) for s in sentences]

    indices = torch.cat(indices, dim=0)

    all_logits = []

    for b in range(input_ids.size(0) // batch_size + 1):
        if style in ["mlm", "mntp"]:
            logits = model(
                input_ids[b * batch_size : (b+1) * batch_size, :].contiguous(),
                attention_mask[b * batch_size : (b+1) * batch_size, :].contiguous(),
            ).logits
            logits = torch.gather(
                logits,
                dim=1,
                index=indices[b * batch_size : (b+1) * batch_size].reshape(-1, 1, 1).expand(-1, -1, logits.size(-1))
            ).squeeze(1)
        if style == "causal":
            logits = model(
                input_ids[b * batch_size : (b+1) * batch_size, :].contiguous(),
            )[0]
            logits = torch.cat([logits[0, :len(sentences[0])], logits[1, :len(sentences[1])]], dim=0)

        all_logits.append(logits)

    logits = torch.cat(all_logits, dim=0)

    return logits, sentences


def calculate_scores_with_temperature(logits: torch.Tensor, sentences: list[list[int]], device: torch.device, temperatures: torch.Tensor | None = None) -> torch.Tensor:
    """This function takes the logits of a pair sentences, the tokenized sentences,
    and the list of temperatures to evaluate the performance of the model.

    Args:
        logits(torch.Tensor): The logits of sentences.
        sentences(list[list[int]]): Tokenized sentences.
        device(torch.device): The device the model is on.
        temperatures(torch.Tensor | None): A tensor of temperatures to test the
            model at. (If None, the model will be evaluated at temperature 1)

    Returns:
        torch.Tensor: A tensor containing the sum logits of the model at each
            temperature.
    """
    if temperatures is None:
        temperatures = torch.ones(1, device=device)

    labels = torch.cat(sentences).unsqueeze(-1).expand(temperatures.size(0), -1, -1).to(device)

    logits = logits.unsqueeze(0) / temperatures.view(-1, 1, 1)
    log_p = F.log_softmax(logits, dim=-1)
    total_score = log_p.gather(index=labels, dim=-1).squeeze(-1)

    scores, offset = [], 0
    for i in range(len(sentences)):
        from_index = offset
        to_index = offset + sentences[i].size(0)
        scores.append(total_score[:, from_index:to_index].sum(-1))
        offset = to_index

    scores = torch.stack(scores, dim=1)

    return scores
