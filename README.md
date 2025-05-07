# 2024 BabyLM Challenge Evaluation Pipeline

![BabyLM Challenge](assets/babylm.png)

## Overview

This code provides the backend for the BabyLM Challenge's evaluation pipeline. This year we decided to implement it from scratch. It currently supports 3 different evaluation types: fine-tuning (sequence), sentence-level zero-shot logits calculations, and word level logits calculations (although the last one is implemented for a specific task).

If you have questions about or suggestions for this code, please open an issue and consider [joining our Slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-2gqgqaumu-5ebxxADuT561aT_ooKbT1Q). Join the `#evaluation` channel, which is dedicated to support for use of this repository.

We also welcome pull requests!

## Install

> [!Note]
> The package is currently not installable given that it is a first version. Instead we recommend installing the packages needed and using it as a python module, i.e. to run a part of the pipeline (for example finetuning) you would to do: `python -m evaluation_pipeline.finetune.run ...` from the root folder (the folder that contains the evaluation_pipeline folder).

To be able to use the pipeline you need to install the requirements.txt packages.

> [!Warning]
> These packages were installed using Python 3.13, in case some of the packages are not compatible with your Python version (either because the version is too recent or is not supported). In that case, you could either update your Python version or pip/conda install the following packages: `transformers`, `torch`, `scikit-learn`, `numpy`, `pandas`, `statsmodels`, and `nltk`.

## Data

Download the `evaluation_data` folder in [this OSF directory](https://osf.io/ryjfm/). Place it in the root directory of this repository.

Due to large file sizes and license restrictions, we do not provide images in the OSF directory. Instead, we link to HuggingFace datasets, two of which require approval (which is immediate). Go to these URLs:
- [Winoground](https://huggingface.co/datasets/facebook/winoground)
- [EWoK](https://huggingface.co/datasets/ewok-core/ewok-core-1.0)

On both pages, make sure you're logged in to your HuggingFace account, and request approval. Then, in your terminal, log in to your account using `huggingface-cli login`, and enter your HuggingFace login token.

For DevBench data, run `devbench/download_data.sh` from the root directory of this repository.

For EWoK data, run `ewok/dl_and_filter.py` from the root directory of this repository.

## Evaluation 
This year, we provide different sets of evaluation tasks for different tracks.

### Text-only evaluation
If you are participating in one of the text-only tracks (Strict or Strict-small), use these instructions.
#### Zero-shot evaluation

Use the following shell script to evaluate on BLiMP:
```
./eval_blimp.sh <path_to_model>
```

And use the following shell script to evaluate on EWoK:
```
./eval_ewok.sh <path_to_model>
```

These should work out-of-the-box if you are using a HuggingFace-based autoregressive model. If you are using a masked language model, change `--model hf` to `--model hf-mlm`. If you are using a custom model not included in HuggingFace's standard architectures list, you'll also need to add the `backend` argument to `--model_args`. To do this, change `--model_args pretrained=$MODEL_NAME` to `--model_args pretrained=$MODEL_NAME,backend="mlm"` if you are using a masked LM, or `backend="causal"` if you are using an autoregressive model.

If you are instead using Mamba or another non-HF model, change the `--model` argument in the script. Use `--model mamba_ssm` for Mamba models, or `--model gguf`/`--model ggml` for Llama.cpp models. (Note that these both require additional dependencies; see Optional Extras below for installation instructions.) See the README of [the original lm-evaluation-harness repository](https://github.com/EleutherAI/lm-evaluation-harness) for a complete list of supported models.

#### Fine-tuning or low-rank adapter training

Like last year, we provide a script to support fine-tuning on all tasks. Running `finetune_model.sh <model_name>`
will fine-tune your model on all (Super)GLUE tasks. You can also optionally specify hyperparameters like batch size,
learning rate, among others.

Here are the hyperparameters used for fine-tuning for all tasks. Feel free to modify these, or to set task-specific hyperparameters:
| Hyperparameter | Value |
| -------------- | ----- |
| Initial learning rate | 5e-5 |
| Batch size | 64 |
| Maximum epochs | 10 |
| Evaluate every (epochs) | 1 |
| Patience | 3 |

This year, we are also providing support for training low-rank adapters instead of full model fine-tuning. This change was motivated by (1) greater compute-efficiency; (2) lower disk space requirements; and (3) modularity. To train low-rank adapters on all (Super)GLUE evaluation tasks, run `train_lora.sh`.

By default, this uses the same hyperparameters for all tasks. Here are the defaults:
| Hyperparameter | Value |
| -------------- | ----- |
| Initial learning rate | 3e-4 |
| Batch size | 64 |
| Maximum epochs | 32 |
| Evaluate every (epochs) | 1 |
| LoRA alpha | 16 |
| LoRA rank | 8 |
| LoRA dropout | 0.1 |

The checkpoint with the best validation performance is the one that is evaluated and saved.

Feel free to modify the hyperparameters, and even to modify the type of adapter or fine-tuning method used. (We have not directly integrated support for QLoRA or ReFT, but we welcome pull requests that add these features!)

### Multimodal evaluation

If you are participating in the multimodal track, use these instructions.

First, run your models on the text-only evaluations, including BLiMP, the BLiMP supplement, EWoK, and (Super)GLUE. As long as your model is compatible with the AutoModelForCausalLM and AutoModelForSequenceClassification classes, you can use the same instructions as above to evaluate on the text-only tasks.

In addition, use the following command to evaluate on Winoground (where we use an unpaired text score) and VQA (accuracy with 7 distractors).
```
./eval_multimodal.sh <path_to_model>
```

Also use the following command to evaluate on DevBench:
```
./eval_devbench.sh <path_to_model> <model_type> (<image_model>)
```
Here, `model_type` refers to the architecture of the model. For example, for `babylm/git-2024`, the model type would be `git`. See the `eval_devbench.sh` script for more information.

## Baselines
The baseline models are available from the BabyLM huggingface page here: https://huggingface.co/babylm . All models for this year's challenge have `-2024` appended to their names.

For the strict and strict-small tracks, we release [BabyLlama](https://aclanthology.org/2023.conll-babylm.24/) and [LTG-BERT](https://aclanthology.org/2023.conll-babylm.20/) baselines. These architectures were chosen because they were the winning methods from last year's challenge. Models containing `-100m` are for the strict track; those containing `-10m` are for strict-small.

For the multimodal tracks, we release [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) and [GIT](https://openreview.net/pdf?id=b4tMhpN0JC) baselines.

Here are scores for each model on each evaluation task. Each task score is an unweighted mean of each subtask score within that task. We also show macroaverages, which are simply means of each task score (i.e., means across a row of the table). NOTE: for GLUE, we average *accuracies* for all tasks except QQP and MRPC (where we use F1 scores), and CoLA (where we use the Matthews correlation coefficient). See end of README for more detailed score breakdowns.

**Strict-small Track (10M)**

| Model | BLiMP | BLiMP Supplement | EWoK | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- |
| BabyLlama | 69.8 | 59.5 | 50.7 | 63.3 | *60.8* |
| LTG-BERT | 60.6 | 60.8 | 48.9 | 60.3 | *57.7* |

The LTG-BERT scores here are lower than expected given that this was last year's winning system. We believe this is because of our choice of hyperparameters---specifically, the number of epochs: we trained all models for approximately 20 epochs. LTG-BERT benefits from training for many more epochs than other models can feasibly train for without overfitting, so perhaps it would perform better with longer training. This is somewhat supported by its results on the Strict track, where the same number of epochs corresponds to many more training steps.

**Strict Track (100M)**

| Model | BLiMP | BLiMP Supplement | EWoK | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- |
| BabyLlama | 73.1 | 60.6 | 52.1 | 69.0 | *63.7* |
| LTG-BERT | 69.2 | 66.5 | 51.9 | 68.4 | *64.0* |

**Multimodal Track**

Here, we show the performance of the Flamingo and GIT baselines on all text-only *and* multimodal tasks. We also show how performance changes on the multimodal tasks when images are not provided to the model during evaluation (i.e., we use the same trained text-and-image model, but modify the evaluation setup to remove any visual information).

| Model | BLiMP | BLiMP Supplement | EWoK | GLUE | *Text Macroaverage* | 
| --- | --- | --- | --- | --- | --- |
| Flamingo | 70.9 | 65.0 | 52.7 | 69.5 | *64.5* |
| GIT | 65.2 | 62.7 | 52.4 | 65.5 | *61.5* |

| Model | Winoground | VQA | DevBench | *Vision Macroaverage* |
| --- | --- | --- | --- | --- |
| Flamingo | 51.6 | 52.3 | 59.5 | *54.5* |
| Flamingo (no vision) | 50.0 | 45.0 | - | *47.5(\*)* |
| GIT | 55.5 | 54.1 | 51.1 | *53.6* |
| GIT (no vision) | 50.0 | 48.4 | - | *49.2(\*)* |

(*) Not directly comparable to other macroaverages, since DevBench scores without vision are not well-defined. These rows are more useful as comparison points for Winoground and VQA with and without visual signals.

## Submission Format
You will upload your models and your models' predictions on the evaluation tasks. You can upload these to OpenReview, or provide links to Google Drive uploads in the OpenReview submission form.

To collect your results into the expected format, use `collect_results.py`. This will produce a Gzipped JSON file containing your model's predictions for all tasks. It also runs a simple verification to make sure you've included predictions for all examples for all tasks.

**If you're submitting to the text-only track, you can use the following to collect your results:**
```
python collect_results.py <name_of_model>
```
Where `<name_of_model>` usually corresponds to the part of your model's name after the final "/" (i.e., the directory name where your results are being saved under `results/<task_name>`). For example, for the `babylm/git-2024` baseline, this would be `git-2024`.

**If you're submitting to the multimodal track, add the `--include_vision_tasks` argument to the `collect_results.py` command.**

If you chose to use LoRA instead of fine-tuning for (Super)GLUE, add the `--glue_lora` argument to the `collect_reuslts.py` command.

Note that you don't have to use this script to collect your results if you've been using an alternate evaluation setup! Also note that the JSON doesn't necessarily need to be Gzipped. Thus, if you'd like to see the exact format you need to put your results into, you may unzip the sample JSONs provided here and inspect them.

### Text-only tasks
For the text-only track, the JSON should contain one entry per task. Each task entry contains separate entries for all subtasks. Each subtask contains a list of dictionaries, where each dictionary has only two fields: an example ID and a prediction value. For example, for GLUE's BoolQ subtask:
```
{"glue": {"boolq": {"predictions": [{"id": "boolq_0", "pred": 0}, {"id": "boolq_1", "pred": 1}, ...]}}}
```

For all other text tasks, the "pred" key takes a string value instead of integers. In the samples provided here, there are spaces at the beginning of the strings, but this will not be required to be scored correct.

### Vision tasks
For VQA and Winoground, the same prediction formatting is used as for the text-only tasks. All predictions should be strings.

For DevBench, "predictions"'s value should be a matrix of floats. This matrix has an exact expected size (which the verification function in `collect_results.py` checks for).

----
----

### Additional Features (copied from EleutherAI README)
Batch size selection can be automated by setting the  ```--batch_size``` flag to ```auto```. This will perform automatic detection of the largest batch size that will fit on your device.

The full list of supported arguments are provided [here](./docs/interface.md), and on the terminal by calling `lm_eval -h`. Alternatively, you can use `lm-eval` instead of `lm_eval`.

> [!Note]
> Just like you can provide a local path to `transformers.AutoModel`, you can also provide a local path to `lm_eval` via `--model_args pretrained=/path/to/model`

> [!Note]
> For tasks unsuitable for direct evaluation — either due risks associated with executing untrusted code or complexities in the evaluation process — the `--predict_only` flag is available to obtain decoded generations for post-hoc evaluation.

If you have a Metal compatible Mac, you can run the eval harness using the MPS back-end by replacing `--device cuda:0` with `--device mps` (requires PyTorch version 2.1 or higher). **Note that the PyTorch MPS backend is still in early stages of development, so correctness issues or unsupported operations may exist. If you observe oddities in model performance on the MPS back-end, we recommend first checking that a forward pass of your model on `--device cpu` and `--device mps` match.**

> [!Note]
> You can inspect what the LM inputs look like by running the following command:
> ```bash
> python write_out.py \
>     --tasks <task1,task2,...> \
>     --num_fewshot 5 \
>     --num_examples 10 \
>     --output_base_path /path/to/output/folder
> ```
> This will write out one text file for each task.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
lm_eval --model openai \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

## Advanced Usage Tips

For models loaded with the HuggingFace  `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library. For example, you can pass a local path via `pretrained=` or use models finetuned with [PEFT](https://github.com/huggingface/peft) by taking the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument:
```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

Models provided as delta weights can be easily loaded using the Hugging Face transformers library. Within --model_args, set the delta argument to specify the delta weights, and use the pretrained argument to designate the relative base model to which they will be applied:
```bash
lm_eval --model hf \
    --model_args pretrained=Ejafa/llama_7B,delta=lmsys/vicuna-7b-delta-v1.1 \
    --tasks hellaswag
```

[GPTQ](https://github.com/PanQiWei/AutoGPTQ) quantized models can be loaded by specifying their file names in `,autogptq=NAME` (or `,autogptq=True` for default names) in the `model_args` argument:

```bash
lm_eval --model hf \
    --model_args pretrained=model-name-or-path,autogptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.

To save evaluation results provide an `--output_path`. We also support logging model responses with the `--log_samples` flag for post-hoc analysis.

Additionally, one can provide a directory with `--use_cache` to cache the results of prior runs. This allows you to avoid repeated execution of the same (model, task) pairs for re-scoring.

For a full list of supported arguments, check out the [interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) guide in our documentation!

## Visualizing Results

You can seamlessly visualize and analyze the results of your evaluation harness runs using both Weights & Biases (W&B) and Zeno.

### Weights and Biases

With the [Weights and Biases](https://wandb.ai/site) integration, you can now spend more time extracting deeper insights into your evaluation results. The integration is designed to streamline the process of logging and visualizing experiment results using the Weights & Biases (W&B) platform.

The integration provide functionalities

- to automatically log the evaluation results,
- log the samples as W&B Tables for easy visualization,
- log the `results.json` file as an artifact for version control,
- log the `<task_name>_eval_samples.json` file if the samples are logged,
- generate a comprehensive report for analysis and visualization with all the important metric,
- log task and cli specific configs,
- and more out of the box like the command used to run the evaluation, GPU/CPU counts, timestamp, etc.

First you'll need to install the lm_eval[wandb] package extra. Do `pip install lm_eval[wandb]`.

Authenticate your machine with an your unique W&B token. Visit https://wandb.ai/authorize to get one. Do `wandb login` in your command line terminal.

Run eval harness as usual with a `wandb_args` flag. Use this flag to provide arguments for initializing a wandb run ([wandb.init](https://docs.wandb.ai/ref/python/init)) as comma separated string arguments.

```bash
lm_eval \
    --model hf \
    --model_args pretrained=microsoft/phi-2,trust_remote_code=True \
    --tasks hellaswag,mmlu_abstract_algebra \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output/phi-2 \
    --limit 10 \
    --wandb_args project=lm-eval-harness-integration \
    --log_samples
```

In the stdout, you will find the link to the W&B run page as well as link to the generated report. You can find an example of this workflow in [examples/visualize-wandb.ipynb](examples/visualize-wandb.ipynb), and an example of how to integrate it beyond the CLI.

### Support

The best way to get support is to open an issue on this repo or join the [BabyLM slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-2gqgqaumu-5ebxxADuT561aT_ooKbT1Q). Join the `#evaluation-pipeline` channel, which is dedicated to support for use of this repository.

## Optional Extras
Extras dependencies can be installed via `pip install -e ".[NAME]"`

| Name          | Use                                   |
|---------------|---------------------------------------|
| anthropic     | For using Anthropic's models          |
| deepsparse     | For running NM's DeepSparse models    |
| dev           | For linting PRs and contributions     |
| gptq          | For loading models with GPTQ          |
| hf_transfer   | For speeding up HF Hub file downloads |
| ifeval        | For running the IFEval task           |
| neuronx       | For running on AWS inf2 instances     |
| mamba         | For loading Mamba SSM models          |
| math          | For running math task answer checking |
| multilingual  | For multilingual tokenizers           |
| openai        | For using OpenAI's models             |
| optimum       | For running Intel OpenVINO models     |
| promptsource  | For using PromptSource prompts        |
| sentencepiece | For using the sentencepiece tokenizer |
| sparseml      | For using NM's SparseML models        |
| testing       | For running library test suite        |
| vllm          | For loading models with vLLM          |
| zeno          | For visualizing results with Zeno     |
|---------------|---------------------------------------|
| all           | Loads all extras (not recommended)    |


## Cite as
Please cite both of the following papers if you use this repository in your work:
```
@article{babylm-2024,
      title={[Call for Papers] The 2nd {BabyLM} {C}hallenge: Sample-efficient pretraining on a developmentally plausible corpus}, 
      author={Leshem Choshen and Ryan Cotterell and Michael Y. Hu and Tal Linzen and Aaron Mueller and Candace Ross and Alex Warstadt and Ethan Wilcox and Adina Williams and Chengxu Zhuang},
      year={2024},
      journal={Computing Research Repository},
      volume={arXiv:2404.06214},
      url={https://arxiv.org/abs/2404.06214}
}

@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 12,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
}
```

## Detailed Score Breakdown

**Strict-small Track (10M)**

*GLUE (Default: Acc.)*
| Model | BoolQ | CoLA (MCC) | MNLI | MNLI-mm | MRPC (F1) | MultiRC | QNLI | QQP (F1) | RTE | SST-2 | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BabyLlama | 65.0 | 2.2 | 72.4 | 74.2 | 82.0 | 60.1 | 82.8 | 83.6 | 49.6 | 86.2 | 38.5 | *63.3* |
| LTG-BERT | 68.8 | 0.0 | 68.9 | 68.9 | 82.2 | 58.5 | 76.5 | 34.2 | 58.3 | 85.1 | 61.5 | *60.3* |


*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |
| BabyLlama | 49.6 | 54.7 | 41.2 | 86.0 | 66.1 | *59.5* |
| LTG-BERT | 54.2 | 62.5 | 49.1 | 79.9 | 58.2 | *60.8* |


*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BabyLlama | 50.5 | 51.7 | 49.4 | 54.2 | 50.4 | 50.6 | 53.5 | 50.7 | 50.3 | 49.8 | 46.7 | *50.7* |
| LTG-BERT | 50.2 | 51.0 | 45.3 | 42.5 | 49.1 | 51.0 | 48.1 | 51.7 | 53.4 | 50.6 | 45.3 | *48.9* |

---
**Strict Track (100M)**

*GLUE (Default: Acc.)*
| Model | BoolQ | CoLA (MCC) | MNLI | MNLI-mm | MRPC (F1) | MultiRC | QNLI | QQP (F1) | RTE | SST-2 | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BabyLlama | 66.1 | 37.3 | 75.6 | 76.2 | 86.8 | 62.1 | 83.1 | 84.5 | 60.4 | 88.3 | 38.5 | *69.0* |
| LTG-BERT | 61.7 | 34.6 | 77.7 | 78.1 | 83.1 | 52.6 | 78.2 | 86.7 | 46.8 | 91.5 | 61.5 | *68.4* |


*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |
| BabyLlama | 45.6 | 56.2 | 44.8 | 83.9 | 72.5 | *60.6* |
| LTG-BERT | 55.0 | 75.0 | 53.3 | 87.5 | 61.4 | *66.5* |


*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BabyLlama | 50.1 | 55.5 | 50.0 | 57.5 | 51.4 | 50.5 | 56.7 | 52.7 | 49.7 | 50.0 | 49.0 | *52.1* | 
| LTG-BERT | 50.1 | 55.8 | 50.6 | 58.3 | 48.9 | 50.9 | 53.8 | 51.4 | 50.8 | 53.8 | 51.4 | 51.2 | 50.8 | 49.2 | *51.9* |

---
**Multimodal Track**

*GLUE (Default: Acc.)*
| Model | BoolQ | CoLA (MCC) | MNLI | MNLI-mm | MRPC (F1) | MultiRC | QNLI | QQP (F1) | RTE | SST-2 | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flamingo | 69.1 | 36.7 | 75.8 | 76.4 | 84.2 | 60.5 | 83.8 | 85.1 | 60.4 | 90.4 | 42.3 | *69.5* |
| GIT | 67.0 | 0.0 | 75.2 | 74.5 | 82.2 | 58.6 | 81.9 | 84.7 | 62.6 | 88.8 | 45.3 | *65.5* |

*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |
| Flamingo | 48.8 | 75.0 | 43.6 | 86.2 | 71.4 | *65.0* |
| GIT | 48.9 | 67.2 | 49.7 | 86.6 | 61.1 | *62.7* |

*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flamingo | 50.8 | 61.0 | 55.3 | 53.3 | 50.9 | 50.1 | 52.5 | 54.4 | 49.4 | 49.6 | 52.2 | *52.7* |
| GIT | 51.0 | 61.9 | 51.2 | 54.2 | 50.2 | 49.9 | 52.5 | 51.4 | 52.4 | 50.2 | 51.6 | *52.4* |


*DevBench*
| Model | THINGS (RSA) | TROG (Acc.) | Visual Vocab (Acc.) | *Macroaverage* |
| --- | --- | --- | --- | --- |
| Flamingo | 46.5 | 51.3 | 80.7 | *59.5* |
| GIT | 32.6 | 38.2 | 82.4 | *51.1* |

| Model | THINGS (RSA) | TROG (Human sim.) | Visual Vocab (Human sim.) | *Macroaverage* |
| --- | --- | --- | --- | --- |
| Flamingo | 46.5 | 47.7 | 75.2 | *56.4* |
| GIT | 32.6 | 44.7 | 75.3 | *50.8* |

The human similarity scores are computed as `exp(-D)`, where `D` is the KL divergence from human response probability distributions to model logits. We exponentiate the negative value to normalize the divergence into a metric within the range [0,1], and to ensure that higher values are better. Note that the macroaverages reported in **Baselines** are from the first table containing accuracies and the THINGS RSA.

Winoground and VQA do not contain subtasks, so scores for these can be found above in the **Baselines** section.
