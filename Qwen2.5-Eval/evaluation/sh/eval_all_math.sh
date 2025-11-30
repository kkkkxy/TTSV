#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES="0"

MODEL_NAME_OR_PATH="model/Qwen2.5-Math-7B"
PROMPT_TYPE="qwen25-math-cot"
DATA_NAMES="math500"
MAX_TOKENS_PER_CALL="3072"


SPLIT="test"
NUM_TEST_SAMPLE=-1

IFS=',' read -ra DATASETS <<< "$DATA_NAMES"
ALL_EXIST=true

TUNING_EMBEDDINGS_PATH="checkpoints/qwen2.5-math-7b_math500"
OUTPUT_DIR="eval_result"
mkdir -p $OUTPUT_DIR

TOKENIZERS_PARALLELISM=false \
python3 -u Qwen2.5-Eval/evaluation/math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAMES} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --batch_size 64 \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --save_outputs \
    --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
    --overwrite \
    --tuning_embeddings_path ${TUNING_EMBEDDINGS_PATH} 
