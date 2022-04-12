#!/usr/bin/env bash
# -*- coding:utf-8 -*-
EXP_ID=$(date +%F-%H-%M-$RANDOM)
export CUDA_VISIBLE_DEVICES="0"
export batch_size="16"
export model_name=model_name
export data_name=data_name
export lr=1e-4
export lr_scheduler=constant_with_warmup
export label_smoothing="0"
export epoch=30
export eval_steps=50
export warmup_steps=0

OPTS=$(getopt -o b:d:m:i:t:k:s:l:f: --long batch:,device:,model:,data:,run-time:,seed:,lr:,lr_scheduler:,label_smoothing:,epoch:,eval_steps:,warmup_steps: -n 'parse-options' -- "$@")

if [ $? != 0 ]; then
  echo "Failed parsing options." >&2
  exit 1
fi

eval set -- "$OPTS"

while true; do
  case "$1" in
  -b | --batch)
    batch_size="$2"
    shift
    shift
    ;;
  -d | --device)
    CUDA_VISIBLE_DEVICES="$2"
    shift
    shift
    ;;
  -m | --model)
    model_name="$2"
    shift
    shift
    ;;
  -i | --data)
    data_name="$2"
    shift
    shift
    ;;
  -k | --run-time)
    run_time="$2"
    shift
    shift
    ;;
  -s | --seed)
    seed="$2"
    shift
    shift
    ;;
  -l | --lr)
    lr="$2"
    shift
    shift
    ;;
  --lr_scheduler)
    lr_scheduler="$2"
    shift
    shift
    ;;
  --label_smoothing)
    label_smoothing="$2"
    shift
    shift
    ;;
  --epoch)
    epoch="$2"
    shift
    shift
    ;;
  --eval_steps)
    eval_steps="$2"
    shift
    shift
    ;;
  --warmup_steps)
    warmup_steps="$2"
    shift
    shift
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "$1" not recognize.
    exit
    ;;
  esac
done

# google/mt5-base -> google_mt5-base

model_name_log=$(echo ${model_name} | sed -s "s/\//_/g")
model_folder=models/score/${data_name}_${EXP_ID}_${model_name_log}
data_folder=./data/${data_name}

export TOKENIZERS_PARALLELISM=false

for index in $(seq 1 ${run_time}); do

  output_dir=${model_folder}_run${index}

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python model/token_classify_model.py \
    --do_train --do_eval --do_predict \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --metric_for_best_model eval_accuracy \
    --load_best_model_at_end \
    --max_seq_length=72 \
    --num_train_epochs=${epoch} \
    --train_file=${data_folder}/train.json \
    --dev_file=${data_folder}/dev.json \
    --test_file=${data_folder}/test.json \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 4)) \
    --output_dir=${output_dir} \
    --logging_dir=${output_dir}_log \
    --learning_rate=${lr} \
    --model_name_or_path=${model_name} \
    --lr_scheduler_type=${lr_scheduler} \
    --label_smoothing_factor=${label_smoothing} \
    --eval_steps ${eval_steps} \
    --warmup_steps ${warmup_steps} \
    --seed=${seed}${index}
done