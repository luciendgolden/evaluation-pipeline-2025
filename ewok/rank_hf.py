import torch
from calculate_logits_scores_hf import calculate_logits, calculate_scores_with_temperature


@torch.no_grad()
def rank(sentences, model, tokenizer, device, batch_size, style, temperatures=None):

    logits, sentences, context_length = calculate_logits(sentences, model, tokenizer, device, batch_size, style)
    scores = calculate_scores_with_temperature(logits, sentences, context_length, device, temperatures)

    ranking = torch.argsort(scores, dim=1, descending=True).tolist()

    return ranking
