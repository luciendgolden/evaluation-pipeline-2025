import torch

from tqdm import tqdm
import argparse
import json
import pathlib
import os

from tokenizers import Tokenizer
# from transformers import AutoModelForMaskedLM, AutoTokenizer

from evaluator import evaluate


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument("--output_dir", default="lambada_results", type=pathlib.Path, help="The output directory where the results will be written.")
    parser.add_argument("--data", default="data/lambada.jsonl", type=pathlib.Path, help="Path to file containing the lambada dataset, we expect it to be in a JSONL format.")
    parser.add_argument("--model_path_or_name", default="baseline/baseline.bin", type=pathlib.Path, help="The local path to the model binary.")
    parser.add_argument("--tokenizer_path", default="../../tokenizer_100M.json", type=str, help="The vocabulary the model was trained on.")
    parser.add_argument("--backend", default="mlm", type=str, help="The evaluation backend strategy.", choices=["mlm", "mntp", "causal", "prefix"])
    parser.add_argument("--config_file", default="../../configs/base.json", type=pathlib.Path)
    parser.add_argument("--architecture", default="base", type=str, help="The architecture of the model, available: base, attglu, attgate, densemod, densesubmod, densecont, elc, qkln")

    # Optinal Parameters
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Outputs the prompt, answer and prediction of the model. Stops after num_prompts prompts.")
    parser.add_argument("--num_prompts", default=10, type=int, help="Number of verbose prompts to output. Only used when verbose is True.")

    args = parser.parse_args()

    return args


def load_config(args):
    with open(args.config_file, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


if __name__ == "__main__":

    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    match args.architecture:
        case "base":
            from model import Bert
        case "extra":
            from model_extra import Bert
        case _:
            raise ValueError(f"The architecture cannot be {args.architecture}, it has to be one of the following: base, extra.")

    with open(args.data, "r") as f:
        new_texts = [json.loads(line) for line in f if len(line.strip()) > 0]

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name)
    args = load_config(args)
    model = Bert(args)

    model.load_state_dict(torch.load(args.model_path_or_name, map_location="cpu", weights_only=True))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(device)
    model.eval()

    verbose = args.verbose
    num_prompts = args.num_prompts

    correct_answers = 0
    total_answers = 0
    perplexity = 0.0
    first_perplexity = 0.0

    progress_bar = tqdm(new_texts)

    for i, text in enumerate(progress_bar):
        answer = text["answer"]
        prompt = text["prompt"]

        prediction, loss, first_loss = evaluate(prompt, answer, tokenizer, model, device, style=args.backend, verbose=verbose)

        perplexity += loss

        if prediction.strip() == answer.strip():
            correct_answers += 1

        total_answers += 1

        accuracy = correct_answers/total_answers * 100.0
        avg_perplexity = torch.exp(perplexity/total_answers)
        avg_first_perplexity = torch.exp(first_perplexity/total_answers)

        progress_bar.set_description(f"Accuracy: {accuracy:.2f}%, Perplexity: {avg_perplexity:.2f}, First Perplexity: {avg_first_perplexity:.2f}")

        if verbose and i == num_prompts:
            break

    print(f"Accuracy: {correct_answers/total_answers * 100.0}")
    print(f"Perplexity: {torch.exp(perplexity/total_answers)}")
    print(f"First Perplexity: {torch.exp(first_perplexity/total_answers)}")

    args.model_name = args.model_path_or_name.stem
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(args.output_dir / args.model_name):
        os.mkdir(args.output_dir / args.model_name)
    if not os.path.exists(args.output_dir / args.model_name / args.backend):
        os.mkdir(args.output_dir / args.model_name / args.backend)
    args.output_path = args.output_dir / args.model_name / args.backend

    if not verbose:
        with (args.output_path / "results.txt").open("w") as file:
            print(f"ACCURACY: {correct_answers/total_answers * 100.0}", file=file)
            print(f"PERPLEXITY: {torch.exp(perplexity/total_answers)}", file=file)
            print(f"FIRST PERPLEXITY: {torch.exp(first_perplexity/total_answers)}", file=file)
