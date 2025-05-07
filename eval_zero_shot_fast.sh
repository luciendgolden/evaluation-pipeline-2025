#!/bin/bash

MODEL_PATH=$1
BACKEND=$2
BLIMP_DIR=${3:-"blimp/data"}
EWOK_DIR=${4:-"ewok/data"}
READING_DIR=${4:-"reading/data"}

python -m evaluation_pipeline.sentence_zero_shot.run_zero_shot --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${BLIMP_DIR}/blimp_sampled" --save_predictions
python -m evaluation_pipeline.sentence_zero_shot.run_zero_shot --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${BLIMP_DIR}/supplemental_sampled" --save_predictions
python -m evaluation_pipeline.sentence_zero_shot.run_zero_shot --model_path_or_name $MODEL_PATH --backend $BACKEND --task ewok --data_path "${EWOK_DIR}/ewok_smapled" --save_predictions
python -m evaluation_pipeline.reading.run --model_path_or_name $MODEL_PATH --backend $BACKEND --data_path "${READING_DIR}/all_measures.csv"