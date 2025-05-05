#!/bin/bash

MODEL_PATH=$1
LR=${2:-3e-5}           # default: 3e-5
BSZ=${4:-32}            # default: 32
MAX_EPOCHS=${5:-10}     # default: 10
SEED=${6:-42}           # default: 42

model_basename=$(basename $MODEL_PATH)

for task in {boolq,mnli,mrpc,qqp,rte,multirc,wsc}; do
        
    python finetune/run.py \
        --model_name_or_path "$MODEL_PATH" \
        --train_data "glue/data/$task.train.jsonl" \
        --valid_data "glue/data/$task.valid.jsonl" \
        --predict_data "glue/data/$task.test.jsonl" \
        --task "$task" \
        --num_labels 2 \
        --batch_size $BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "./results" \
        --save \
        --save_dir "./models" \
        --metrics accuracy f1 \
        --metric_for_valid accuracy \
        --seed $SEED \
        --verbose
done