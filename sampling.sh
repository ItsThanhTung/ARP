export CODE_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/code
export DATA_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/data
# export PRETRAINED_PATH=exp_real_data_1.5_prompt/model-80000
# export PRETRAINED_PATH=runwayml/stable-diffusion-v1-5
export PRETRAINED_PATH=exp_24k_data_1.5/model-35000
export OUT_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/sampling_data/test_24k

GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"

CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 sampling.py \
                                    --pretrained_model_name_or_path=$PRETRAINED_PATH \
                                    --output_dir=$OUT_DIR \
                                    --dataloader_num_workers=16 \
                                    --allow_tf32 \
                                    --mixed_precision=no \
                                    --enable_xformers_memory_efficient_attention \
                                    --num_samples=500 \

python calculate_fid.py
