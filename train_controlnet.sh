export CODE_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/code
export DATA_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/data
export PRETRAINED_PATH=exp_real_data_1.5_prompt/model-80000
export CONTROLNET_PATH=lllyasviel/control_v11p_sd15_seg


GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"

CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                    -m controlnet.train \
                                    --pretrained_model_name_or_path=$PRETRAINED_PATH \
                                    --controlnet_model_name_or_path=$CONTROLNET_PATH \
                                    --dataset_file=$DATA_DIR/real_train_semantic_data.txt \
                                    --output_dir="exp_controlnet_real_data_1.5_prompt" \
                                    --resolution=512 \
                                    --train_batch_size=8 \
                                    --num_train_epochs=250 \
                                    --checkpointing_steps=1000 \
                                    --checkpoints_total_limit=10 \
                                    --gradient_accumulation_steps=8 \
                                    --gradient_checkpointing \
                                    --learning_rate=2e-06 \
                                    --scale_lr \
                                    --lr_scheduler=constant_with_warmup \
                                    --lr_warmup_steps=500 \
                                    --dataloader_num_workers=16 \
                                    --allow_tf32 \
                                    --report_to=tensorboard \
                                    --validate_file=$DATA_DIR/real_test_semantic_data.txt \
                                    --validation_steps=1000 \
                                    --mixed_precision=no \
                                    --tracker_project_name="train_controlnet" \
                                    --enable_xformers_memory_efficient_attention \