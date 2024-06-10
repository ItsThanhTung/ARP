export CODE_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/code
export DATA_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/data
export PRETRAINED_PATH=exp_real_data_1.5_prompt/model-80000
export CONTROLNET_PATH=exp_controlnet_real_data_1.5_prompt/checkpoint-18000/controlnet
export OUT_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/sampling_data/test_real

GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"

CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 sampling_controlnet.py \
                                    --pretrained_model_name_or_path=$PRETRAINED_PATH \
                                    --controlnet_model_name_or_path=$CONTROLNET_PATH \
                                    --output_dir=$OUT_DIR \
                                    --dataset_file=$DATA_DIR/real_test_semantic_data.txt \
                                    --dataloader_num_workers=16 \
                                    --allow_tf32 \
                                    --mixed_precision=no \
                                    --enable_xformers_memory_efficient_attention \
                                    --num_samples=4 \