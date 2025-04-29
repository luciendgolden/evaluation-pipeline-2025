from __future__ import annotations

import torch
from collections import defaultdict, Counter
from tqdm import tqdm
from rank import rank
from read_file import read_file

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from argparse import Namespace


def _compute_accuracy(counts: dict[str, dict[str, Counter]]) -> dict[str, list[dict[str, float]]]:
    accuracies = dict()
    for subdomain, count in counts.items():
        accuracies[subdomain] = [
            {
                key: count["correct"][i][key] / count["total"][i][key] * 100.0
                for key in count["correct"][i].keys()
                }
            for i in range(len(count["correct"]))
            ]

    average_accuracies = [sum(accuracies["UID"][i].values()) / len(accuracies["UID"][i].values()) for i in range(len(accuracies["UID"]))]

    return accuracies, average_accuracies


@torch.no_grad()
def evaluate_sentence(model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase, device: torch.device, args: Namespace) -> tuple[list[float], dict[str, list[dict[str, float]]], list[float], None | list[dict[str, dict[str, list[dict[str, str]]]]]]:
    """This function calculates the sentence logprobs of a correct and incorrect
    sentence. Compares them and gives which, a given model, is more likely. Then
    it returns the accuracy of the model over the full data and a more fine-tuned
    evalution based on the task given.

    Args:
        model(torch.nn.Module): The model to evaluate (this would be a HuggingFace
            model).
        tokenizer(PreTrainedTokenizerBase): The tokenizer which the model is
            trained on.
        device(torch.device): The device the model is on.
        args(Namespace): A class containing all the arguments such as the data
            path, task, etc.

    Returns:
        tuple[list[float], dict[str, list[dict[str, float]]],
            list[float],
            None | list[dict[str, dict[str, list[dict[str, str]]]]]]:
            A 4 item tuple containing the temperatures evaluated, the finegrained
            accuracies, the average accuracies, and
            the predictions of the model (optional)
    """
    if args.max_temperature is None:
        temperatures = torch.ones(1, device=device) * args.min_temperature
    else:
        temperatures = torch.arange(args.min_temperature, args.max_temperature + args.temperature_interval, args.temperature_interval, device=device).clamp(min=1e-6)

    pairs = read_file(args.data_path, args.task)

    counts = dict()
    final_predictions = []

    for key in pairs[0].keys():
        if key in ["sentences", "label"]:
            continue

        counts[key] = {"correct": [Counter() for _ in range(temperatures.size(0))], "total": [Counter() for _ in range(temperatures.size(0))]}

    if args.predict:
        all_predictions = [defaultdict(list) for _ in range(len(temperatures))]
        uid_counter = [Counter() for _ in range(len(temperatures))]

    progress_bar = tqdm(total=len(pairs))

    for pair in pairs:
        finegrained_ranking = rank(pair["sentences"], model, tokenizer, device, args.batch_size, args.backend, temperatures=temperatures)

        for i, ranking in enumerate(finegrained_ranking):
            for subdomain, count in counts.items():
                if ranking[0] == pair["label"]:
                    count["correct"][i][pair[subdomain]] += 1
                count["total"][i][pair[subdomain]] += 1
                if args.predict:
                    uid = pair["UID"]
                    all_predictions[i][uid].append({"id": f"{uid}_{uid_counter[i][uid]}", "pred": " " + (pair["sentences"][ranking[0]])})

            if args.predict:
                uid_counter[i][uid] += 1

        progress_bar.update()

    if args.predict:
        for i in range(len(temperatures)):
            temp_pred = dict()
            for k, v in all_predictions[i].items():
                temp_pred[k] = dict()
                temp_pred[k]["predictions"] = v
            final_predictions.append(temp_pred)

    accuracies, average_accuracies = _compute_accuracy(counts)

    return temperatures, accuracies, average_accuracies, final_predictions
