import torch
import torch.nn.functional as F


def create_attention_mask(input_ids, padding, style="bidirectional", device="cpu"):
    attention_mask = None

    if style == "bidirectional":
        attention_mask = torch.ones_like(input_ids, dtype=torch.int, device=device)
        attention_mask[:, attention_mask.size(-1) - padding:] = 0

    return attention_mask


@torch.no_grad()
def calculate_logits(sentences, model, tokenizer, device, batch_size, style):
    mask_index = tokenizer.mask_token_id
    pad_index = tokenizer.pad_token_id
    cls_index = torch.tensor([tokenizer.cls_token_id])
    sep_index = torch.tensor([tokenizer.sep_token_id])

    sentences = [torch.tensor(tokenizer(s, add_special_tokens=False).input_ids) for s in sentences]

    def prepare_mlm(tokens, padding: int, device: str | torch.device = "cpu"):
        tokens = torch.cat([cls_index, tokens, sep_index, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - 2 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[1:-(1 + padding), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = create_attention_mask(input_ids, padding, device=device)
        return input_ids, attention_mask

    def prepare_mntp(tokens, padding: int, device: str | torch.device = "cpu"):
        tokens = torch.cat([cls_index, tokens, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - 1 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[1:(-padding if padding > 0 else None), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = create_attention_mask(input_ids, padding, device=device)
        return input_ids, attention_mask

    def prepare_causal(tokens, padding: int, device: str | torch.device = "cpu"):
        input_ids = torch.cat([cls_index, tokens, torch.full((padding,), fill_value=pad_index)]).to(device)
        return input_ids

    max_length = max(s.size(0) for s in sentences)
    if style == "mlm":
        input_ids, attention_masks = zip(*[prepare_mlm(s, max_length - s.size(0), device) for s in sentences])
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_masks, dim=0)
        indices = [torch.arange(1, 1 + len(s), device=device) for s in sentences]
    elif style == "causal":
        input_ids = [prepare_causal(s, max_length - s.size(0), device) for s in sentences]
        input_ids = torch.stack(input_ids, dim=0)
        indices = [torch.arange(0, len(s), device=device) for s in sentences]
    elif style == "mntp":
        input_ids, attention_masks = zip(*[prepare_mntp(s, max_length - s.size(0), device) for s in sentences])
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


def calculate_scores_with_temperature(logits, sentences, device, temperatures=None):

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
