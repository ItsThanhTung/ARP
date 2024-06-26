export CODE_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/code
# export DATA_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/synthetic_test_data.json
export DATA_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_no_none_semantic_latent_data.json
# export PRETRAINED_PATH=exp_real_data_1.5_prompt/model-80000
export PRETRAINED_PATH=runwayml/stable-diffusion-v1-5
# export CONTROLNET_PATH=exp_real_20K_synthetic_no_none_semantic_latent_data/checkpoint-33000/controlnet


GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"

export CONTROLNET_PATH=/lustre/scratch/client/vinai/users/tungdt33/ARP/code/ARP/exp_real_20K_synthetic_no_none_semantic_latent_data/checkpoint-42000/controlnet
export OUT_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/sampling_data/exp_real_20K_synthetic_no_none_semantic_latent_data_real
CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
                                    --rdzv-backend=c10d --rdzv-endpoint=localhost:0 sampling_controlnet.py \
                                    --pretrained_model_name_or_path=$PRETRAINED_PATH \
                                    --controlnet_model_name_or_path=$CONTROLNET_PATH \
                                    --output_dir=$OUT_DIR \
                                    --dataset_file=$DATA_DIR \
                                    --dataloader_num_workers=16 \
                                    --allow_tf32 \
                                    --mixed_precision=no \
                                    --enable_xformers_memory_efficient_attention \
                                    --num_samples=2 \
                                    
python calculate_fid.py  --data_root=$OUT_DIR/images >> log.txt