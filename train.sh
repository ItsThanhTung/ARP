export CODE_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/code
export DATA_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/train_6k.json
export PRETRAINED_PATH=stabilityai/stable-diffusion-2-1-base

GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"


CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                         dreambooth/train_dreambooth.py \
                                        --pretrained_model_name_or_path=$PRETRAINED_PATH \
                                        --instance_data_dir=$DATA_DIR \
                                        --instance_prompt='' \
                                        --output_dir="exp_debug" \
                                        --resolution=512 \
                                        --center_crop \
                                        --train_batch_size=8 \
                                        --sample_batch_size=1 \
                                        --num_train_epochs=10 \
                                        --checkpointing_steps=400 \
                                        --checkpoints_total_limit=200 \
                                        --gradient_accumulation_steps=1 \
                                        --gradient_checkpointing \
                                        --learning_rate=5e-07 \
                                        --scale_lr \
                                        --lr_scheduler=constant_with_warmup \
                                        --lr_warmup_steps=100 \
                                        --dataloader_num_workers=16 \
                                        --allow_tf32 \
                                        --report_to=tensorboard \
                                        --num_validation_images=4 \
                                        --validation_steps=1000 \
                                        --mixed_precision=no \
                                        --enable_xformers_memory_efficient_attention \
                                        --use_ema \
                                        --validation_prompt='roads building sky person vegetation car' \
                                        # --resume_from_checkpoint /lustre/scratch/client/vinai/users/tungdt33/ARP/code/ARP/exp_synthetic_data_1.5/checkpoint-10000