export DEBUG_MODE="true"

DATA_ROOT="/path/to/videos/root"
DATA_FILE_PATHS="datasets/RL-VQA_training_failure_prediction.jsonl"
MODEL_NAME_OR_PATH="ckpt/VQR1-7B-YouTubeUGC"

SRC_DIR="src/open-r1-multimodal"
RUN_NAME="LoRA_training_failure_prediction"
export LOG_PATH="output/$RUN_NAME/log_$RUN_NAME.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    $SRC_DIR/src/open_r1/grpo_jsonl.py \
    --deepspeed $SRC_DIR/local_scripts/zero2.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --question_template only_scoring \
    --dataset_name YouTubeUGC \
    --image_folders $DATA_ROOT \
    --data_file_paths $DATA_FILE_PATHS \
    --max_prompt_length 1024 \
    --num_generations 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 10 \
    --run_name $RUN_NAME \
    --save_steps 10 \
    --save_only_model true \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --reward_funcs "accuracy_loss_difficult" "format_loss_difficult"