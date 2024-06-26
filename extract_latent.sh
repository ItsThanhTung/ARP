export CODE_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/code/ARP
# export PRETRAINED_PATH=stabilityai/stable-diffusion-2-1-base
export PRETRAINED_PATH=runwayml/stable-diffusion-v1-5
export DATA_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/data/train_fix.json
export OUT_DIR=/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_24K/latent_1.5
export OUT_PATH=/lustre/scratch/client/vinai/users/tungdt33/ARP/data/train_fix_latent.json

GPU_STRING=$1
GPU_COUNT=$(echo $GPU_STRING | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo $GPU_STRING | cut -d ',' -f 1)
echo "Number of GPUs: $GPU_COUNT - First GPU: $FIRST_GPU - Available GPU: $GPU_STRING"


# CUDA_VISIBLE_DEVICES=$GPU_STRING torchrun --nnodes 1 --nproc_per_node $GPU_COUNT \
#                                     --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
#                                          dreambooth/extract_latent.py \
#                                         --pretrained_model_name_or_path=$PRETRAINED_PATH \
#                                         --instance_data_dir=$DATA_DIR \
#                                         --output_dir=$OUT_DIR \
#                                         --train_batch_size=16 \
#                                         --dataloader_num_workers=4 \
python preprocess_data/add_latent_to_json.py  --input_path=$DATA_DIR --output_path=$OUT_PATH --output_dir=$OUT_DIR
