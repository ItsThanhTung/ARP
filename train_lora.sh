export CODE_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/code
export DATA_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/data/MASKED_REAL/train_latent
export PRETRAINED_PATH=stabilityai/stable-diffusion-2-1-base

GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"


CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                         dreambooth/train_dreambooth_lora.py \
                                        --pretrained_model_name_or_path=$PRETRAINED_PATH \
                                        --instance_data_dir=$DATA_DIR \
                                        --instance_prompt='' \
                                        --output_dir="exp_real_data_2.1_lora" \
                                        --resolution=512 \
                                        --center_crop \
                                        --train_batch_size=16 \
                                        --sample_batch_size=1 \
                                        --num_train_epochs=100 \
                                        --checkpointing_steps=5000 \
                                        --checkpoints_total_limit=10 \
                                        --gradient_accumulation_steps=4 \
                                        --gradient_checkpointing \
                                        --learning_rate=2e-06 \
                                        --scale_lr \
                                        --lr_scheduler=constant_with_warmup \
                                        --lr_warmup_steps=500 \
                                        --dataloader_num_workers=16 \
                                        --allow_tf32 \
                                        --report_to=tensorboard \
                                        --validation_prompt='roads building sky person vegetation car' \
                                        --num_validation_images=4 \
                                        --validation_epochs=1 \
                                        --mixed_precision=no \
                                        --rank=32 \
                                        --enable_xformers_memory_efficient_attention \
                                        # --resume_from_checkpoint /home/ubuntu/Workspace/tung-dev/DGInStyle/exp_base_all_data_2.1/checkpoint-15000