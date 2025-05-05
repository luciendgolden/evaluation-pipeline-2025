# File: run_zero_shot.py
# ----------------------

import os
import pathlib
import json
import pickle
import argparse
import gc
from tqdm import tqdm
from _io import TextIOWrapper

import math
from collections import Counter, defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from dataset import get_dataloader
from compute_results import compute_results

def _parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_path", default="", type=pathlib.Path, help="Path to the data directory")
    parser.add_argument("--output_dir", default="results", type=pathlib.Path, help="Path to the data directory")
    parser.add_argument("--task", default="blimp", type=str, help="The task that is being evaluated.",
                        choices=["blimp", "ewok", "comps", "entity_tracking"])

    parser.add_argument("--model_path_or_name", default="ltg/gpt-bert-babylm-small", type=str, help="Path to the model to evaluate.")
    parser.add_argument("--backend", default="mlm", type=str, help="The evaluation backend strategy", choices=["mlm", "causal", "mntp"])    

    parser.add_argument("--min_temperature", default=1.0, type=float, help="Minimum temperature to apply to the logits.")
    parser.add_argument("--max_temperature", default=None, type=float, help="Maximum temperature to apply to the logits. If None, onlny the minimum temperature will be considered.")
    parser.add_argument("--temperature_interval", default=0.05, type=float, help="Step size between temperatures applied to the logits.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for evaluation")
    parser.add_argument("--non_causal_batch_size", default=64, type=int, help="Mini-batch size to process each batch of inputs involving masked tokens")
    parser.add_argument("--save_predictions", action="store_true", help="Whether or not to save predictions.")

    return parser.parse_args()

def get_model(args: argparse.ArgumentParser):
    if args.backend in ["mlm", "mntp"]:
        model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    elif args.backend == "causal":
        model = AutoModelForCausalLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()

    return model
    
def get_temperatures(args: argparse.ArgumentParser):
    if args.max_temperature is None:
        temperatures = torch.ones(1) * args.min_temperature
    else:
        temperatures = torch.arange(
            args.min_temperature,
            args.max_temperature + args.temperature_interval,
            args.temperature_interval,
        ).clamp(min=1e-6)
    return temperatures.tolist()

def process_results(args: argparse.ArgumentParser, results: dict):
    """This function computes accuracy metrics and, if necessary, other dataset-specific metrics
    given dataset sizes and numbers of correct predictions

    Args:
        args (argparse.ArgumentParser): ArgumentParser object used to determine task
        results (dict): Results obtained from running compute_results
    """
    # Compute accuracies
    accuracies = {temp : {} for temp in results}
    for temp, temp_results in results.items():
        for subdomain, count_dict in temp_results.items():
            keys = count_dict["total"].keys()
            subdomain_accs = {key : 100.0 * count_dict["correct"][key] / count_dict["total"][key] for key in keys}
            accuracies[temp][subdomain] = subdomain_accs
            
    # Average accuracies
    average_accuracies = {}
    if args.task in ["blimp", "ewok"]:
        for temp, temp_results in results.items():
            total = 0
            correct = 0
            for subdomain, count_dict in temp_results.items():
                for key in count_dict["total"].keys():
                    correct += count_dict["correct"][key]
                    total += count_dict["total"][key]
                break
            average_accuracies[temp] = 100 * correct / total
    elif args.task == "comps":
        for temp, subdomain_dict in accuracies.items():
            accs = [v for _, v in subdomain_dict["split"].items()]
            average_accuracies[temp] = sum(accs) / len(accs)
    else:
        splits = ["regular", "ambiref", "move_contents"]
        for temp, subdomain_dict in accuracies.items():
            split_accs = []
            split_dict = subdomain_dict["subset"]
            for split in splits:
                split_keys = [key for key in split_dict if key.startswith(split)]
                curr_acc = sum([split_dict[key] for key in split_keys]) / len(split_keys)

                split_dict[split] = curr_acc
                split_accs.append(curr_acc)
            average_accuracies[temp] = sum(split_accs) / len(split_accs)

    return accuracies, average_accuracies

def create_evaluation_report(temperature: float, avg_accuracy: torch.Tensor, accuracies: dict[str, list[dict[str, float]]], file: TextIOWrapper | None = None) -> None:
    """This function creates a report and either saves it to a file or prints it to the terminal.

    Args:
        temperature(float): The temperature at which the model is evaluated.
        temperature_pos(int): The position of the evaluated temperature.
        avg_accuracy(torch.Tensor): The average accuracy of the model at the given temperature.
        avg_accuracy(dict[str, list[dict[str, float]]]): The finegrained accuracies of the model
            at the given temperature.
        file(TextIOWrapper | None): The file to write to results to. (If None, it will printed
            printed to the terminal)
    """
    print(f"TEMPERATURE: {temperature:.2f}", file=file)
    print(file=file)

    for domain, accuracy in accuracies.items():
        print(f"### {domain.upper()} ACCURACY", file=file)
        for subdomain, acc in accuracy.items():
            print(f"{subdomain}: {acc:.2f}", file=file)
        print(file=file)

    print("### AVERAGE ACCURACY", file=file)
    print(f"{avg_accuracy:.2f}", file=file)
    print(file=file)

def save_predictions(args, predictions, best_temp):
    temp_predictions = predictions[best_temp]
    flattened_predictions = []
    for _, prediction in temp_predictions.items():
        flattened_predictions.append(prediction)

    with (args.output_path / "predictions_at_best_temperature.json").open("w") as f:
        json.dump(flattened_predictions, f)

def main():
    args = _parse_arguments() 
    dataset = args.data_path.stem
    args.model_name = pathlib.Path(args.model_path_or_name).stem
    args.output_path = args.output_dir / args.task / args.model_name / dataset / args.backend
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Get results
    model = get_model(args) 
    dataloader = get_dataloader(args)
    temperatures = get_temperatures(args)
    results, predictions = compute_results(args, model, dataloader, temperatures)

    # Process results 
    accuracies, average_accuracies = process_results(args, results)
    best_acc = -1
    best_temp = -1
    for temperature, acc in average_accuracies.items():
        print(f"{temperature}\t{acc:.2f}")
        if acc > best_acc:
            best_acc = acc
            best_temp = temperature
    print()

    # Report and save
    create_evaluation_report(best_temp, average_accuracies[best_temp], accuracies[best_temp])
    with (args.output_path / "best_temperature_report.txt").open("w") as f:
        create_evaluation_report(best_temp, average_accuracies[best_temp], accuracies[best_temp], file=f)

    # Save predictions
    if args.save_predictions:
        save_predictions(args, predictions, best_temp)

if __name__ == "__main__":
    main()
