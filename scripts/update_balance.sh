export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python ../src/update_selector.py \
    --use_deepspeed false \
    --checkpoint_path /data/yimin/peft/TASA/selector/exp6/selector_6_1/checkpoint-230 \
    --model_name_or_path /data/yimin/models/base/Qwen/Qwen2-7B \
    --update_config_path /data/yimin/peft/TASA/config/update_config.json \
    --output_dir /data/yimin/peft/TASA/selector/exp6/selector_6_1_update_2 \
    --option balance \
    --option_type mix \
    --reduce_ratio 0.1 \
    --do_train True \
    --use_instruction false \
    --use_output false \
    --bf16 true \
    --fp16 false \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 4e-4 \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 512 \