from __future__ import annotations

from evaluate import evaluate_sentence
import argparse
import pathlib
import torch
from write_to_file import create_evaluation_report
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import json


def _parse_arguments():
    parser = argparse.ArgumentParser("Parser of argument ")

    # Required parameters
    parser.add_argument("--data_path", default="../ewok/data/ewok_filtered", type=pathlib.Path, help="Path to data directory.")
    parser.add_argument("--output_dir", default="results", type=pathlib.Path, help="The output directory where the results will be written.")
    parser.add_argument("--task", default="ewok", type=str, help="The task that is being evaluated.", choices=["blimp", "ewok"])
    parser.add_argument("--model_path_or_name", default="ltg/gpt-bert-babylm-small", type=str, help="Path to the model to evaluate.")
    parser.add_argument("--backend", default="mlm", type=str, help="The evaluation backend strategy", choices=["mlm", "causal", "mntp"])
    parser.add_argument("--min_temperature", default=1.0, type=float, help="Minimum temperature to apply to the logits.")
    parser.add_argument("--max_temperature", default=None, type=float, help="Maximum temperature to apply to the logits. If None, only the minimum temperature will be considered.")
    parser.add_argument("--temperature_interval", default=0.05, type=float, help="Step size between temperatures applied to the logits.")
    parser.add_argument("--batch_size", default=64, type=int, help="Number of sentences to see at the same time. Only important for MLM and MNTP, where a sentence is generated for each possible mask position.")
    parser.add_argument("--predict", action=argparse.BooleanOptionalAction, default=False, help="Whether or not to save predictions.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = args.data_path.stem
    args.model_name = pathlib.Path(args.model_path_or_name).stem
    args.output_path = args.output_dir / args.task / args.model_name / dataset / args.backend
    args.output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, trust_remote_code=True)

    if args.backend in ["mlm", "mntp"]:
        model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    elif args.backend == "causal":
        model = AutoModelForCausalLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(device)
    model.eval()

    temperatures, accuracies, average_accuracies, final_predictions = evaluate_sentence(model, tokenizer, device, args)

    for temperature, acc in zip(temperatures.tolist(), average_accuracies):
        print(f"{temperature}\t{acc:.2f}")
    print()

    average_accuracies = torch.tensor(average_accuracies)
    max_temp = torch.argmax(average_accuracies)
    max_temperature = temperatures[max_temp]

    create_evaluation_report(max_temperature, max_temp, average_accuracies[max_temp], accuracies)

    with (args.output_path / "best_temperature_report.txt").open("w") as f:
        create_evaluation_report(max_temperature, max_temp, average_accuracies[max_temp], accuracies, file=f)

    if args.predict:
        with (args.output_path / "predictions_at_best_temperature.json").open("w") as f:
            json.dump(final_predictions[max_temp], f)
