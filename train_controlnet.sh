export CODE_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/code
export DATA_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/data
# export PRETRAINED_PATH=runwayml/stable-diffusion-v1-5
export PRETRAINED_PATH=exp_6k/model-10000
export CONTROLNET_PATH=lllyasviel/control_v11p_sd15_seg


GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"

CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
                                    -m controlnet.train \
                                    --pretrained_model_name_or_path=$PRETRAINED_PATH \
                                    --dataset_file=/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real \
                                    --output_dir="exp_6k_controlnet" \
                                    --resolution=512 \
                                    --train_batch_size=8 \
                                    --num_train_epochs=250 \
                                    --checkpointing_steps=1000 \
                                    --checkpoints_total_limit=10 \
                                    --gradient_accumulation_steps=8 \
                                    --gradient_checkpointing \
                                    --learning_rate=5e-07 \
                                    --scale_lr \
                                    --lr_scheduler=constant_with_warmup \
                                    --lr_warmup_steps=500 \
                                    --dataloader_num_workers=16 \
                                    --allow_tf32 \
                                    --report_to=tensorboard \
                                    --validate_file=/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP \
                                    --validation_steps=100 \
                                    --mixed_precision=no \
                                    --tracker_project_name="train_controlnet" \
                                    --enable_xformers_memory_efficient_attention \
