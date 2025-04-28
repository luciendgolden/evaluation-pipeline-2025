from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from evaluation_functions import get_p2_mntp, get_p2, get_p2_mlm
from tqdm import tqdm
import pandas as pd
import argparse
import pathlib
import statsmodels.formula.api as smf
from functools import partial
import math
import json


def parse_args():
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument("--output_dir", default="results", type=pathlib.Path, help="The output directory where the results will be written.")
    parser.add_argument("--data", default="data/all_measures.csv", type=pathlib.Path, help="Path to file containing the lambada dataset, we expect it to be in a JSONL format.")
    parser.add_argument("--model_path_or_name", default="ltg/gpt-bert-babylm-small", type=pathlib.Path, help="The path/name to/of the huggingface folder/repository.")
    parser.add_argument("--backend", default="causal", type=str, help="The evaluation backend strategy.", choices=["mlm", "mntp", "causal"])
    parser.add_argument("--number_of_mask_tokens_to_append", default=3, type=int, help="When using either mlm or mntp, the number of mask tokens to append to approximate causal generation.")

    args = parser.parse_args()

    args.output_dir /= args.model_path_or_name.stem
    args.output_dir /= args.backend
    if args.backend in ["mlm", "mntp"]:
        args.output_dir /= f"{args.number_of_mask_tokens_to_append}_mask_tokens"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.data, dtype={'item': str})
    df["item"] = df["item"].fillna("None")

    if args.backend == "causal":
        model = AutoModelForCausalLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.model_path_or_name, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name, trust_remote_code=True)

    if args.backend == "causal":
        p2_function = get_p2
    elif args.backend == "mlm":
        p2_function = partial(get_p2_mlm, num_mask_tokens=args.number_of_mask_tokens_to_append)
    else:
        p2_function = partial(get_p2_mntp, num_mask_tokens=args.number_of_mask_tokens_to_append)

    out = []
    for index, row in tqdm(df.iterrows(), total=len(df)):

        p, _ = p2_function(row["item"], row["word"], model, tokenizer)
        out.append(p)

    p2 = [-math.log(p) for p in out]

    df["pred"] = p2

    pred_file = args.output_dir / "prediction.jsonl"
    with pred_file.open("w") as fj:
        for index, row in df.iterrows():
            print(json.dumps({"Index": index, "Sentence": row["item"], "Word": row["word"], "Logprob": row["pred"]}), file=fj)

    variables = ['RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound', 'self_paced_reading_time',  'ELAN', 'LAN', 'N400', 'P600', 'EPNP', 'PNP']

    correlations = df[["pred"] + variables].corr()["pred"]

    corr_file = args.output_dir / "correlations.txt"
    with corr_file.open("w") as f:
        for index, values in correlations.items():
            if index != "pred":
                print(f"{index}\t{values:.4f}", file=f)

    results = []

    for dv in variables:
        # baseline model
        temp = df[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
        # first fit baseline model without predictability
        OLS_baseline = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length', data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        aic_baseline = float(OLS_baseline.aic)
        temp = df[["pred", dv, "Subtlex_log10", "length", "context_length"]]
        # experimental model with iv
        OLS_model = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred', data=temp).fit()
        is_sig = float(OLS_model.tvalues["pred"])
        the_p = float(OLS_model.pvalues["pred"])
        the_B = float(OLS_model.params["pred"])
        R2_model = float(OLS_model.rsquared)
        aic_model = float(OLS_model.aic)
        results.append({
            "Predicted variable": dv,
            "Coefficient": the_B,
            "Number of standard deviations": is_sig,
            "P-value": the_p,
            "R2": R2_model,
            "Change in R2 from baseline": R2_model-R2_baseline,
            "AIC": aic_model,
            "Change in AIC from baseline": aic_model-aic_baseline,
        })

    predictability_file = args.output_dir / "predictive_power.jsonl"
    with predictability_file.open("w") as fj:
        for res in results:
            print(json.dumps(res), file=fj)
