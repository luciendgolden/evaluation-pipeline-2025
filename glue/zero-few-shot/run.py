import argparse
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from evaluate import evaluate
from prompt_styles import prompt_styles_dict
import pathlib
import os
import json
import torch
from statistics import mean, stdev


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--evaluation_dataset", default="../data/sst2.valid.jsonl", type=pathlib.Path, help="Path to GLUE task to evaluate on.")
    parser.add_argument("--output_dir", default="../glue_results", type=pathlib.Path, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_path_or_name", default="ltg/gpt-bert-babylm-small", type=str, help="Path to a previous checkpointed training state.")
    parser.add_argument("--backend", default="mntp", type=str, help="The evaluation backend strategy", choices=["causal", "mntp", "prefix"])
    parser.add_argument("--labels", default=["negative", "positive"], nargs="+", type=str, help="The labels for each class (the order reflects the class so the first label corresponds to class 0)")
    parser.add_argument("--num_shots", default=0, type=int, help="The number of in-context shots to feed the model")
    parser.add_argument("--few_shot_dataset", default=None, type=pathlib.Path, help="The dataset to choose the in-context examples from.")
    parser.add_argument("--num_repetitions", default=1, type=int, help="Number of times to repeat the run (useful for few-shot).")
    parser.add_argument("--task", default="sst2", type=str, help="The GLUE task being evaluated.")
    parser.add_argument("--prompt_style_name", default="default", type=str, help="The name of the prompt style to use (combined witht he task name).")
    parser.add_argument("--delimiter", default="/n/n", type=str, help="How to seperate between examples and the prompt.")

    args = parser.parse_args()

    return args


def write_report(y_true, y_pred, file=None):
    accs, f1s, mccs = [], [], []
    for yt, yp in zip(y_true, y_pred):
        accs.append(accuracy_score(y_true=yt, y_pred=yp)*100)
        f1s.append(f1_score(y_true=yt, y_pred=yp))
        mccs.append(matthews_corrcoef(y_true=yt, y_pred=yp))

    for i, (acc, f1, mcc) in enumerate(zip(accs, f1s, mccs)):
        print("#"*30, f"RUN {i+1}", "#"*30, file=file)
        print(f"Accuracy: {acc:.2f}\t\tF1: {f1:.2f}\t\tMCC: {mcc:.2f}", file=file)
        print(file=file)

    print("#"*35, "AVERAGE", "#"*35, file=file)
    print(f"Accurcay: {mean(accs):.2f} ± {stdev(accs) if len(accs) > 2 else 0:.2f}\t\tF1: {mean(f1s):.2f} ± {stdev(f1s) if len(f1s) > 2 else 0:.2f}\t\tMCC: {mean(mccs):.2f} ± {stdev(mccs) if len(mccs) > 2 else 0:.2f}", file=file)


if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.model_name = pathlib.Path(args.model_path_or_name).stem
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(args.output_dir / args.model_name):
        os.mkdir(args.output_dir / args.model_name)
    if not os.path.exists(args.output_dir / args.model_name / args.task):
        os.mkdir(args.output_dir / args.model_name / args.task)
    if not os.path.exists(args.output_dir / args.model_name / args.task / "zero-few-shot"):
        os.mkdir(args.output_dir / args.model_name / args.task / "zero-few-shot")
    if not os.path.exists(args.output_dir / args.model_name / args.task / "zero-few-shot" / args.backend):
        os.mkdir(args.output_dir / args.model_name / args.task / "zero-few-shot" / args.backend)
    args.output_path = args.output_dir / args.model_name / args.task / "zero-few-shot" / args.backend

    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, trust_remote_code=True)

    if args.backend == "mntp":
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    elif args.backend in ["causal", "prefix"]:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(device)
    model.eval()

    dataset = []
    with args.evaluation_dataset.open("r") as file:
        for line in file.readlines():
            dataset.append(json.loads(line))

    few_shot_dataset = None
    if args.few_shot_dataset is not None:
        few_shot_dataset = []
        with args.few_shot_dataset.open("r") as file:
            for line in file.readlines():
                few_shot_dataset.append(json.loads(line))

    style = "_".join([args.task, args.prompt_style_name])
    if style in prompt_styles_dict:
        prompt_style = prompt_styles_dict[style]
    else:
        print(f"WARNING: The prompt style {style} does not exist, defaulting to [text]\\n[label]!")
        prompt_style = "{0}\n{1}"
    y_true, y_pred = evaluate(dataset, prompt_style, model, tokenizer, args, few_shot_dataset)

    write_report(y_true, y_pred)
    with (args.output_path / f"{args.num_shots}-shots.txt").open("w") as file:
        write_report(y_true, y_pred, file)
