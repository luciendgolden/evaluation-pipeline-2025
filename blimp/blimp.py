import argparse
import torch
import json
from collections import Counter
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import wandb
import os
from tqdm import tqdm
import pathlib
from collections import defaultdict

from rank import rank
from tokenizers import Tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_path", default="data/blimp", type=pathlib.Path, help="Path to BLiMP data.")
    parser.add_argument("--output_dir", default="blimp_results", type=pathlib.Path, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--tokenizer_path", default="../../models/tokenizer_100M.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--model_path_or_name", default="../lambada/baseline/baseline.bin", type=pathlib.Path, help="Path to a previous checkpointed training state.")
    parser.add_argument("--config_file", default="../../configs/base.json", type=pathlib.Path)
    parser.add_argument("--backend", default="mlm", type=str, help="The evaluation backend strategy", choices=["mlm", "causal", "mntp"])
    parser.add_argument("--min_temperature", default=1.0, type=float, help="Minimum temperature to apply to the logits.")
    parser.add_argument("--max_temperature", default=None, type=float, help="Maximum temperature to apply to the logits. If None, only the minimum temperature will be considered.")
    parser.add_argument("--temperature_interval", default=0.05, type=float, help="Step size between temperatures applied to the logits.")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--architecture", default="base", type=str, help="The architecture of the model.", choices=["base", "extra"])
    parser.add_argument("--predict", action=argparse.BooleanOptionalAction, default=False, help="Whether or not to save predictions.")

    args = parser.parse_args()

    return args


def load_config(args):
    with open(args.config_file, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def create_report(temperature, avg_accuracy, field_accuracy, linguistics_term_accuracy, uid_accuracy, file=None):
    print(f"TEMPERATURE: {temperature:.2f}", file=file)
    print(file=file)

    print("### FIELD ACCURACY", file=file)
    for key in field_accuracy.keys():
        print(f"{key}: {field_accuracy[key]:.2f}", file=file)
    print(file=file)

    print("### LINGUISTIC TERM ACCURACY", file=file)
    for key in linguistics_term_accuracy.keys():
        print(f"{key}: {linguistics_term_accuracy[key]:.2f}", file=file)
    print(file=file)

    print("### UID ACCURACY", file=file)
    for key in uid_accuracy.keys():
        print(f"{key}: {uid_accuracy[key]:.2f}", file=file)
    print(file=file)

    print("### AVERAGE ACCURACY", file=file)
    print(f"{avg_accuracy:.2f}", file=file)
    print(file=file)


@torch.no_grad()
def evaluate(model, tokenizer, device, args):
    if args.max_temperature is None:
        temperatures = torch.ones(1, device=device) * args.min_temperature
    else:
        temperatures = torch.arange(args.min_temperature, args.max_temperature + args.temperature_interval, args.temperature_interval, device=device).clamp(min=1e-6)

    field_count = {"correct": [Counter() for _ in range(temperatures.size(0))], "total": [Counter() for _ in range(temperatures.size(0))]}
    uid_count = {"correct": [Counter() for _ in range(temperatures.size(0))], "total": [Counter() for _ in range(temperatures.size(0))]}
    linguistics_term_count = {"correct": [Counter() for _ in range(temperatures.size(0))], "total": [Counter() for _ in range(temperatures.size(0))]}

    if args.predict:
        all_predictions = [defaultdict(list) for _ in range(len(temperatures))]

    # iterate through all .jsonl files in ./data/ directory
    for filename in os.listdir(args.input_path):
        if not filename.endswith(".jsonl"):
            continue

        if args.predict:
            counter = 0

        # open file
        with open(os.path.join(args.input_path, filename), "r") as file:
            # iterate through each line in file
            for line in tqdm(file):
                # parse line
                line = json.loads(line.strip())

                # add to pairs
                if "field" in line:
                    pair = {
                        "good": line["sentence_good"],
                        "bad": line["sentence_bad"],
                        "field": line["field"],
                        "UID": line["UID"],
                        "linguistics_term": line["linguistics_term"]
                    }
                    if pair["field"] == "syntax_semantics":
                        pair["field"] = "syntax/semantics"
                else:
                    pair = {
                        "good": line["sentence_good"],
                        "bad": line["sentence_bad"],
                        "field": "supplemental",
                        "UID": filename.split(".")[0],
                        "linguistics_term": "supplemental"
                    }

                # rank
                finegrained_ranking = rank([pair["good"], pair["bad"]], model, tokenizer, device, args.batch_size, args.backend, temperatures=temperatures)

                for i, ranking in enumerate(finegrained_ranking):
                    if ranking[0] == 0:
                        field_count["correct"][i][pair["field"]] += 1
                        uid_count["correct"][i][pair["UID"]] += 1
                        linguistics_term_count["correct"][i][pair["linguistics_term"]] += 1
                    field_count["total"][i][pair["field"]] += 1
                    uid_count["total"][i][pair["UID"]] += 1
                    linguistics_term_count["total"][i][pair["linguistics_term"]] += 1
                    if args.predict:
                        all_predictions[i][pair["UID"]].append({"id": f"{pair['UID']}_{counter}", "pred": " " + (pair["good"] if ranking[0] == 0 else pair["bad"])})

                if args.predict:
                    counter += 1

    if args.predict:
        final_predictions = []
        for i in range(len(temperatures)):
            temp_pred = dict()
            for k, v in all_predictions[i].items():
                temp_pred[k] = dict()
                temp_pred[k]["predictions"] = v
            final_predictions.append(temp_pred)

    # compute accuracy

    field_accuracy = [{key: field_count["correct"][i][key] / field_count["total"][i][key] * 100.0 for key in field_count["correct"][i].keys()} for i in range(len(finegrained_ranking))]
    uid_accuracy = [{key: uid_count["correct"][i][key] / uid_count["total"][i][key] * 100.0 for key in uid_count["correct"][i].keys()} for i in range(len(finegrained_ranking))]
    linguistics_term_accuracy = [{key: linguistics_term_count["correct"][i][key] / linguistics_term_count["total"][i][key] * 100.0 for key in linguistics_term_count["correct"][i].keys()} for i in range(len(finegrained_ranking))]

    average_accuracies = [sum(uid_accuracy[i].values()) / len(uid_accuracy[i].values()) for i in range(len(finegrained_ranking))]

    for temperature, acc in zip(temperatures.tolist(), average_accuracies):
        print(f"{temperature}\t{acc:.2f}")
    print()

    average_accuracies = torch.tensor(average_accuracies)
    max_temp = torch.argmax(average_accuracies)
    max_temperature = temperatures[max_temp]

    create_report(max_temperature, average_accuracies[max_temp], field_accuracy[max_temp], linguistics_term_accuracy[max_temp], uid_accuracy[max_temp])

    with (args.output_path / "best_temperature_report.txt").open("w") as f:
        create_report(max_temperature, average_accuracies[max_temp], field_accuracy[max_temp], linguistics_term_accuracy[max_temp], uid_accuracy[max_temp], file=f)

    if args.predict:
        with (args.output_path / "predictions_at_best_temperature.json").open("w") as f:
            json.dump(final_predictions[max_temp], f)


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add to this for different models
    match args.architecture:
        case "base":
            from model import Bert
        case "extra":
            from model_extra import Bert
        case _:
            raise ValueError(f"The architecture cannot be {args.architecture}, it has to be one of the following: base, extra.")

    task = args.input_path.stem
    args.model_name = args.model_path_or_name.stem
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(args.output_dir / args.model_name):
        os.mkdir(args.output_dir / args.model_name)
    if not os.path.exists(args.output_dir / args.model_name / task):
        os.mkdir(args.output_dir / args.model_name / task)
    if not os.path.exists(args.output_dir / args.model_name / task / args.backend):
        os.mkdir(args.output_dir / args.model_name / task / args.backend)
    args.output_path = args.output_dir / args.model_name / task / args.backend

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    args = load_config(args)
    model = Bert(args)

    model.load_state_dict(torch.load(args.model_path_or_name, map_location="cpu"))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(device)
    model.eval()

    evaluate(model, tokenizer, device, args)
