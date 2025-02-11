import torch
import torch.nn.functional as F


# Modify this to your style of attention mask, here False is attended while True is ignored
def create_attention_mask(input_ids, style="bidirectional", prefix_length=None, device="cpu"):
    attention_mask = None

    if style == "bidirectional":
        attention_mask = torch.ones_like(input_ids, dtype=torch.int, device=device)
    elif style == "causal":
        attention_mask = None
    elif style == "prefix":
        attention_mask = torch.zeros_like(input_ids, dtype=torch.int)
        attention_mask[:, :prefix_length] = 1

    return attention_mask


@torch.no_grad()
def evaluate(prompt, answer, tokenizer, model, device, style="mlm", verbose=False):
    # Change the values of the tokes to your tokenizer values
    cls_token = tokenizer.cls_token_id
    mask_token = tokenizer.mask_token_id
    sep_token = tokenizer.sep_token_id

    prompt, ending = prompt.split("{answer}")

    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Answer: {answer}")
        print(f"Ending: {ending}")

    inputs = tokenizer(prompt.strip(), add_special_tokens=False).input_ids
    prefix_length = len(inputs) + 1
    ending = tokenizer(f'#{ending.strip()}', add_special_tokens=False).input_ids[1:]
    gold_output = tokenizer(answer, add_special_tokens=False).input_ids

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

    logits = model(inputs, mask)[0][0, -(gold_output.size(1) + len(ending) + 1):-(len(ending) + 1)]
    first_loss = F.cross_entropy(logits[0], gold_output[0][0])
    loss = F.cross_entropy(logits, gold_output[0])

    prediction = tokenizer.decode(logits.argmax(-1).tolist())

    if verbose:
        print(f"Prediction: {prediction}")
        print()
        if prediction.strip() != answer.strip():
            print(f"Wrong answer: {prediction} != {answer}")

    return prediction, loss, first_loss
