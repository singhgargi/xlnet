#!/bin/bash

#### local path
RACE_DIR=data/RACE
INIT_CKPT_DIR=xlnet_cased_L-24_H-1024_A-16
PROC_DATA_DIR=proc_data/race
MODEL_DIR=experiment/race

python run_race.py \
  --use_tpu=False \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --model_config_path=${INIT_CKPT_DIR}/xlnet_config.json \
  --spiece_model_file=${INIT_CKPT_DIR}/spiece.model \
  --output_dir=${PROC_DATA_DIR} \
  --init_checkpoint=${INIT_CKPT_DIR}/xlnet_model.ckpt \
  --model_dir=${MODEL_DIR} \
  --data_dir=${RACE_DIR} \
  --max_seq_length=512 \
  --max_qa_length=128 \
  --uncased=False \
  --do_train=True \
  --train_batch_size=4 \
  --do_eval=True \
  --eval_batch_size=32 \
  --train_steps=85000 \
  --save_steps=5000 \
  --iterations=1000 \
  --warmup_steps=1000 \
  --learning_rate=2e-5 \
  --weight_decay=0 \
  --adam_epsilon=1e-6 \
  $@
