import torch
import torch.nn.functional as F


# Modify this to your style of attention mask, here False is attended while True is ignored
def create_attention_mask(input_ids, style="bidirectional", prefix_length=None, device="cpu"):
    attention_mask = None

    if style == "bidirectional":
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    elif style == "causal":
        attention_mask = torch.ones(input_ids.size(1), input_ids.size(1), dtype=torch.bool, device=device).triu(diagonal=1).unsqueeze(0)
    elif style == "prefix":
        attention_mask = torch.ones(input_ids.size(1), input_ids.size(1), dtype=torch.bool).triu(diagonal=1).to(device).unsqueeze(0)
        attention_mask[:, :prefix_length, :prefix_length] = False

    return attention_mask


@torch.no_grad()
def evaluate(prompt, answer, tokenizer, model, device, style="mlm", verbose=False):
    # Change the values of the tokes to your tokenizer values
    cls_token = tokenizer.token_to_id("[CLS]")
    mask_token = tokenizer.token_to_id("[MASK]")
    sep_token = tokenizer.token_to_id("[SEP]")

    prompt, ending = prompt.split("{answer}")

    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Ending: {ending}")

    inputs = tokenizer.encode(prompt.strip(), add_special_tokens=False).ids
    prefix_length = len(inputs) + 1
    ending = tokenizer.encode(f'#{ending.strip()}', add_special_tokens=False).ids[1:]
    gold_output = tokenizer.encode(answer, add_special_tokens=False).ids

    if style == "mlm":
        inputs = [cls_token] + inputs + [mask_token] * len(gold_output) + ending + [sep_token]
    elif style == "mntp":
        inputs = [cls_token] + inputs + [mask_token] * len(gold_output) + ending
    elif style in ["causal", "prefix"]:
        inputs = [cls_token] + inputs + gold_output + ending

    gold_output = torch.tensor([gold_output]).to(device)

    inputs = torch.tensor(inputs).unsqueeze(0).to(device)

    if verbose:
        print(f"Prompt: {inputs}")
        print(f"Ending: {ending}")
        print(f"Answer: {gold_output}")

    if style in ["mlm", "mntp"]:
        mask = create_attention_mask(inputs, style="bidirectional", device=device)
    elif style == "causal":
        mask = create_attention_mask(inputs, style="causal", device=device)
    elif style == "prefix":
        mask = create_attention_mask(inputs, style="prefix", prefix_length=prefix_length, device=device)

    logits = model(inputs.t(), mask).transpose(0, 1)[0, -(gold_output.size(1) + len(ending) + 1):-(len(ending) + 1)]
    first_loss = F.cross_entropy(logits[0], gold_output[0][0])
    loss = F.cross_entropy(logits, gold_output[0])

    prediction = tokenizer.decode(logits.argmax(-1).tolist())

    if verbose:
        print(f"Prediction: {prediction}")
        print()
        if prediction.strip() != answer.strip():
            print(f"Wrong answer: {prediction} != {answer}")

    return prediction, loss, first_loss
