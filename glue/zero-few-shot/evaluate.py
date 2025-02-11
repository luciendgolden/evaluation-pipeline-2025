import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def evaluate_one(text, prompt_style, model, tokenizer, labels, style="mntp", shots=10, few_shot_dataset=None, delimiter="\n\n"):
    if few_shot_dataset is not None and shots != 0:
        idx = np.random.choice(len(few_shot_dataset), replace=False, size=shots)
    else:
        idx = []

    tokens = [tokenizer(label, add_special_tokens=False).input_ids[0] for label in labels]

    text_shot = []
    for i in idx:
        label = labels[few_shot_dataset[i]['label']]
        text_shot.append(prompt_style.format(few_shot_dataset[i]['sentence'], " " + label))

    if style == "mntp":
        text_shot.append(prompt_style.format(text, tokenizer.mask_token))
    elif style in ["causal", "prefix"]:
        text_shot.append(prompt_style.format(text, ""))

    text = delimiter.join(text_shot)
    encoded_text = tokenizer(text, return_tensors='pt')
    if style == "mntp":
        prediction = model(encoded_text.input_ids, encoded_text.attention_mask)[0][:, :-1]
    elif style == "causal":
        prediction = model(encoded_text.input_ids)[0]
    elif style == "prefix":
        prediction = model(encoded_text.input_ids, encoded_text.attention_mask)[0]

    logits = [prediction[:, -1, token] for token in tokens]
    return np.argmax(logits)


def evaluate_dataset(dataset, prompt_style, model, tokenizer, args, few_shot_dataset=None):
    y_true = []
    y_pred = []
    progress_bar = tqdm(total=len(dataset))
    for i, data in enumerate(dataset):
        pred = evaluate_one(data["sentence"], prompt_style, model, tokenizer, args.labels, style=args.backend, shots=args.num_shots, few_shot_dataset=few_shot_dataset, delimiter=args.delimiter)
        y_true.append(data["label"])
        y_pred.append(pred)
        progress_bar.update()
    progress_bar.close()

    return y_true, y_pred


def evaluate(dataset, prompt_style, model, tokenizer, args, few_shot_dataset=None):
    if args.num_shots == 0:
        num_rep = 1
    else:
        num_rep = args.num_repetitions
        if few_shot_dataset is None:
            print("WARNING: No few-shot dataset was passed! The run will revert to zero-shot.")
            args.num_shots = 0
            num_rep = 1

    y_trues = []
    y_preds = []
    for _ in range(num_rep):
        yt, yp = evaluate_dataset(dataset, prompt_style, model, tokenizer, args, few_shot_dataset)
        y_trues.append(yt)
        y_preds.append(yp)

    return y_trues, y_preds
