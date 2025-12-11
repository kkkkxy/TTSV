#!/bin/bash

accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision 'bf16' \
  --dynamo_backend 'no' \
  train.py \
  --model_name Qwen2.5-Math-1.5B \
  --model_path model/Qwen2.5-Math-1.5B \
  --train_dataset_name 'math500' \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --num_epochs 20 \
  --log_steps 1 \
  --run_name qwen2.5-math-1.5b_math500 \
  --save_root 'checkpoints' \
  --prefix_length 20


