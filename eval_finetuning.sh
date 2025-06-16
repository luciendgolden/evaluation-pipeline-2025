#!/bin/bash
set -e

MODEL_PATH=$1
LR=${2:-3e-5}           # default: 3e-5
BSZ=${3:-32}            # default: 32
BIG_BSZ=${4:-16}        # default: 16
MAX_EPOCHS=${4:-10}     # default: 10
WSC_EPOCHS=${5:-30}     # default: 30
SEED=${5:-42}           # default: 42

model_basename=$(basename $MODEL_PATH)

for task in {boolq,multirc}; do
    echo "Running $task finetuning..."
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BIG_BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid accuracy \
        --seed $SEED \
        --verbose \
        --wandb \
        --wandb_project "BabyLM2025-finetuning" \
        --wandb_entity luciendgolden-university-of-vienna
    echo "$task finetuning completed."
done

echo "Running RTE finetuning..."
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/rte.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/rte.valid.jsonl" \
    --task rte \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose \
    --wandb \
    --wandb_project "BabyLM2025-finetuning" \
    --wandb_entity luciendgolden-university-of-vienna
echo "RTE finetuning completed."

echo "Running WSC finetuning..."
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/wsc.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/wsc.valid.jsonl" \
    --task wsc \
    --num_labels 2 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $WSC_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose \
    --wandb \
    --wandb_project "BabyLM2025-finetuning" \
    --wandb_entity luciendgolden-university-of-vienna
echo "WSC finetuning completed."

for task in {mrpc,qqp}; do
    echo "Running $task finetuning..."
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid f1 \
        --seed $SEED \
        --verbose \
        --wandb \
        --wandb_project "BabyLM2025-finetuning" \
        --wandb_entity luciendgolden-university-of-vienna
    echo "$task finetuning completed."
done

echo "Running MNLI finetuning..."
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path "$MODEL_PATH" \
    --train_data "evaluation_data/full_eval/glue_filtered/mnli.train.jsonl" \
    --valid_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl" \
    --predict_data "evaluation_data/full_eval/glue_filtered/mnli.valid.jsonl" \
    --task mnli \
    --num_labels 3 \
    --batch_size $BSZ \
    --learning_rate $LR \
    --num_epochs $MAX_EPOCHS \
    --sequence_length 512 \
    --results_dir "results" \
    --save \
    --save_dir "models" \
    --metrics accuracy \
    --metric_for_valid accuracy \
    --seed $SEED \
    --verbose \
    --wandb \
    --wandb_project "BabyLM2025-finetuning" \
    --wandb_entity luciendgolden-university-of-vienna
echo "MNLI finetuning completed."

echo "All finetuning tasks completed successfully."