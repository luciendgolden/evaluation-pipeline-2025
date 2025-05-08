# 2025 BabyLM Challenge Evaluation Pipeline

![BabyLM Challenge](assets/babylm.png)

## Overview

This code provides the backend for the BabyLM Challenge's evaluation pipeline. This year we decided to implement it from scratch. It currently supports 3 different evaluation types: fine-tuning (sequence), sentence-level zero-shot logits calculations, and word level logits calculations (although the last one is implemented for a specific task).

A new addition this year is that we have two evaluation types: *fast* evaluation uses a smaller set of evaluation samples, and allows for quick testing of your models, and *full* evaluation can be ran on your final model.

If you have questions about or suggestions for this code, please open an issue and consider [joining our Slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-2gqgqaumu-5ebxxADuT561aT_ooKbT1Q). Join the `#evaluation` channel, which is dedicated to support for use of this repository.

We also welcome pull requests!

## Install

> [!Note]
> The package is currently not installable given that it is a first version. Instead we recommend installing the packages needed and using it as a python module, i.e. to run a part of the pipeline (for example finetuning) you would to do: `python -m evaluation_pipeline.finetune.run ...` from the root folder (the folder that contains the evaluation_pipeline folder).

To be able to use the pipeline you need to install the `requirements.txt` packages.

> [!Warning]
> These packages were installed using Python 3.13, in case some of the packages are not compatible with your Python version (either because the version is too recent or is not supported). In that case, you could either update your Python version or pip/conda install the following packages: `transformers`, `torch`, `scikit-learn`, `numpy`, `pandas`, `statsmodels`, `datasets`, and `nltk`.

## File Structure

```
evaluation_pipeline
├── __init__.py
├── ewok
│   ├── dl_and_filter.py
│   └── vocab.txt
├── finetune
│   ├── README.md
│   ├── __init__.py
│   ├── classifier_model.py
│   ├── dataset.py
│   ├── run.py
│   ├── trainer.py
│   └── utils.py
├── reading
│   ├── README.md
│   ├── __init__.py
│   ├── evaluation_functions.py
│   └── run.py
└── sentence_zero_shot
    ├── README.md
    ├── __init__.py
    ├── compute_results.py
    ├── dataset.py
    ├── read_files.py
    └── run.py
```

## Data

Download the `evaluation_data` folder in [this OSF directory](https://osf.io/ryjfm/). Place it in the root directory of this repository.

Due to large file sizes and license restrictions, we do not provide images in the OSF directory of the evaluation tasks for the multimodal track. Instead, we link to HuggingFace datasets, two of which require approval (which is immediate). Go to this URL to download this dataset:
- [Winoground](https://huggingface.co/datasets/facebook/winoground)

Furthermore, the EWoK data requires agreeing to the terms & conditions on the HuggingFace Hub, which can be agreed to here:
- [EWoK](https://huggingface.co/datasets/ewok-core/ewok-core-1.0)

On both pages, make sure you're logged in to your HuggingFace account, and request approval. Then, in your terminal, log in to your account using `huggingface-cli login`, and enter your HuggingFace login token.

For DevBench data, run `devbench/download_data.sh` from the root directory of this repository.

For EWoK data, run `python -m evaluation_pipeline.ewok.dl_and_filter` from the root directory of this repository.

For the fast EWoK data, we provide a password-protected ZIP file called `ewok_fast.zip`.

## Evaluation 
This year, we provide different sets of evaluation tasks for different tracks.

### Text-only evaluation
If you are participating in one of the text-only tracks (Strict or Strict-small) or interaction track, use these instructions.
#### Zero-shot evaluation

Use the following shell script to evaluate on the full zero-shot evaluations:
```bash
./eval_zero_shot.sh <path_to_model> <architecture (causal/mntp/mlm)> <eval_dir (optional, default:evaluation_data/full_eval)>
```

Use the following shell script to evaluate on the fast zero-shot evaluations:
```bash
./eval_zero_shot.sh <path_to_model> <revision_name> <architecture (causal/mntp/mlm)> <eval_dir (optional, default:evaluation_data/fast_eval)>
```

> [!Note]
> The revision name indicates the checkpoint to use (for example in the gpt-bert baselines `chck_1M` is the model trained for about 1M words).

These will work out of the box if you use a HuggingFace-based model. In the case you are not, you can either go to the `hf_conversion_tutorial` folder to create a HF repository or adapt the code to work with a pure PyTorch implementation (it should not be too complicated). The implementation currently only supports three types of trained langauge modeling tasks: causal, mlm, and mntp (mlm shifted similarly to causal). If another objective (like diffusion for example) was used to train the models, you will need to edit the files.

#### Fine-tuning or low-rank adapter training

Like last year, we provide a script to support fine-tuning on all tasks:
```bash
./eval_finetune.sh <path_to_model> <learning_rate (optional, default: 3e-5)> <batch_size (optional, default: 32)> <max_epochs (optional, default: 10)> <seed (optional, default: 42)>
```
This will fine-tune your model on all (Super)GLUE tasks.

> [!Note]
> The hyperparameters are shared through all tasks, if you want to have different ones for every task, you will either need to edit the file or run the python command found in the file from the terminal.

> [!Note]
> There are more hyperparameters you can play with! Checkout the README in the finetune folder of the evaluation_pipeline for more information. In addition, you can edit also edit the classifier head.

Here are the hyperparameters used for fine-tuning for all tasks. Feel free to modify these, or to set task-specific hyperparameters:
| Hyperparameter | Value |
| -------------- | ----- |
| Initial learning rate | 5e-5 |
| Batch size | 32 |
| Maximum epochs | 10 |
| Seed | 42 |

### Multimodal evaluation

If you are participating in the multimodal track, use these instructions.

First, run your models on the text-only evaluations, including BLiMP, the BLiMP supplement, EWoK, and (Super)GLUE. As long as your model is compatible with the AutoModelForCausalLM and AutoModelForSequenceClassification classes, you can use the same instructions as above to evaluate on the text-only tasks.

In addition, use the following command to evaluate on Winoground (where we use an unpaired text score) and VQA (accuracy with 7 distractors).
> [!Note]
> Currently under construction.

## Baselines
The baseline models are available from the BabyLM Community huggingface page here: https://huggingface.co/BabyLM-community .

For the strict and strict-small tracks, we release [BabyLlama](https://aclanthology.org/2023.conll-babylm.24/) and [LTG-BERT](https://aclanthology.org/2023.conll-babylm.20/) baselines. These architectures were chosen because they were the winning methods from last year's challenge. Models containing `-100m` are for the strict track; those containing `-10m` are for strict-small.

For the multimodal tracks, we release [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) and [GIT](https://openreview.net/pdf?id=b4tMhpN0JC) baselines.

Here are scores for each model on each evaluation task. Each task score is an unweighted mean of each subtask score within that task. We also show macroaverages, which are simply means of each task score (i.e., means across a row of the table). NOTE: for GLUE, we average *accuracies* for all tasks except QQP and MRPC (where we use F1 scores). See end of README for more detailed score breakdowns.

> [!Note]
> The evaluations are run on the final model (the one trained for 10 epochs (100M words in Strict-small and 1B words in Strict and Interaction)).

**Strict-small Track (10M)**

*Causal*

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | **71.66** | **63.21** | 49.49 | **9.89** | **3.45** | | 43.00 |
|GPT-BERT (Mixed) | 69.62 | 61.56 | **50.23** | 9.50 | 3.37 | | 45.00 |
|GPT-BERT (Masked-focus) | 65.22 | 59.49 | 49.47 | 9.52 | 3.44 | | **68.00** |

*MNTP/MLM*

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | 69.07 | **64.33** | 49.62 | 9.47 | **3.48** | | 43.00 |
|GPT-BERT (Mixed) | **71.29** | 63.30 | 49.93 | **9.78** | 3.33 | | 16.00 |
|GPT-BERT (Masked-focus) | 70.36 | 63.71 | **49.95** | 9.40 | 3.37 | | **57.5** |


**Strict Track (100M)**

*Causal*

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | **79.29** | **70.42** | **52.32** | 8.36 | 3.02 | | 43.5 |
|GPT-BERT (Mixed) | 78.37 | 69.23 | 51.79 | 8.74 | **3.59** | | 39.5 |
|GPT-BERT (Masked-focus) | 74.56 | 63.63 | 51.57 | **8.80** | 3.30 | | **59.00** |

*MNTP/MLM*

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|GPT-BERT (Causal-focus) | | | | 9.08 | 3.12 | | 37.5 |
|GPT-BERT (Mixed) | | | | 9.15 | **3.43** | | 37.00 |
|GPT-BERT (Masked-focus) | | | | **9.34** | 3.34 | | **55.00** |


**Interaction Track**

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

**Multimodal Track**

Here, we show the performance of the Flamingo and GIT baselines on all text-only *and* multimodal tasks. We also show how performance changes on the multimodal tasks when images are not provided to the model during evaluation (i.e., we use the same trained text-and-image model, but modify the evaluation setup to remove any visual information).

| Model | BLiMP | BLiMP Supplement | EWoK | Reading (Eye Tracking) | Reading (Self-Paced Reading Time) | Entity Tracking | WUGs | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

| Model | Winoground | VQA | DevBench | *Vision Macroaverage* |
| --- | --- | --- | --- | --- |
| Flamingo | 51.6 | 52.3 | 59.5 | *54.5* |
| Flamingo (no vision) | 50.0 | 45.0 | - | *47.5(\*)* |
| GIT | 55.5 | 54.1 | 51.1 | *53.6* |
| GIT (no vision) | 50.0 | 48.4 | - | *49.2(\*)* |

(*) Not directly comparable to other macroaverages, since DevBench scores without vision are not well-defined. These rows are more useful as comparison points for Winoground and VQA with and without visual signals.

## Submission Format
> [!Note]
> To Be Announced!

----
## Visualizing Results

You can seamlessly visualize and analyze the results of your evaluation harness runs using Weights & Biases (W&B).

### Weights and Biases

> [!Note]
> Currently we do not support Weights and Biases support, this will be added in the near future.

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
@misc{charpentier2025babylmturns3papers,
      title={BabyLM Turns 3: Call for papers for the 2025 BabyLM workshop}, 
      author={Lucas Charpentier and Leshem Choshen and Ryan Cotterell and Mustafa Omer Gul and Michael Hu and Jaap Jumelet and Tal Linzen and Jing Liu and Aaron Mueller and Candace Ross and Raj Sanjay Shah and Alex Warstadt and Ethan Wilcox and Adina Williams},
      year={2025},
      eprint={2502.10645},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.10645}, 
}
```

## Detailed Score Breakdown

**Strict-small Track (10M)**

*GLUE (Default: Acc.)*
| Model | BoolQ | CoLA (MCC) | MNLI | MNLI-mm | MRPC (F1) | MultiRC | QNLI | QQP (F1) | RTE | SST-2 | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |


*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

---
**Strict Track (100M)**

*GLUE (Default: Acc.)*
| Model | BoolQ | CoLA (MCC) | MNLI | MNLI-mm | MRPC (F1) | MultiRC | QNLI | QQP (F1) | RTE | SST-2 | WSC | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


*BLiMP Supplement (Acc.)*
| Model | Hypernym | QA Congruence (easy) | QA Congruence (tricky) | Subj.-Aux. Inversion | Turn Taking | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |


*EWoK (Acc.)*
| Model | Agent Properties | Material Dynamics | Material Properties | Physical Dynamics | Physical Interactions | Physical Relations | Quantitative Properties | Social Interactions | Social Properties | Social Relations | Spatial Relations | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

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

## Bibliography


