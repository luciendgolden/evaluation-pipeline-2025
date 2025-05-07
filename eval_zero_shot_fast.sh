#!/bin/bash

MODEL_PATH=$1
REVISION_NAME=${2:-"main"}
BACKEND=$3
BLIMP_DIR=${4:-"blimp/data"}
EWOK_DIR=${5:-"ewok/data"}
READING_DIR=${6:-"reading/data"}

python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${BLIMP_DIR}/blimp_sampled" --save_predictions --revision_name $REVISION_NAME
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${BLIMP_DIR}/supplemental_sampled" --save_predictions --revision_name $REVISION_NAME
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task ewok --data_path "${EWOK_DIR}/ewok_sampled" --save_predictions --revision_name $REVISION_NAME
python -m evaluation_pipeline.reading.run --model_path_or_name $MODEL_PATH --backend $BACKEND --data_path "${READING_DIR}/reading_data.csv" --revision_name $REVISION_NAME