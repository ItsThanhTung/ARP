# CUDA_VISIBLE_DEVICES=0 nohup python dreambooth/generate_blip2_captions.py \
                                # --input_path /lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_latent_data.json \
                                # --output_path /lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_latent_data_with_prompt.json >> real_latent_data.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python dreambooth/generate_blip2_captions.py \
#                                 --input_path /lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/synthetic_train_latent_data.json \
#                                 --output_path /lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/synthetic_train_latent_data_with_prompt.json >> synthetic_train_latent_data.txt &
CUDA_VISIBLE_DEVICES=2 nohup python dreambooth/generate_blip2_captions.py \
                                --input_path /lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/synthetic_test_data.json \
                                --output_path /lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/synthetic_test_data_with_prompt.json >> synthetic_test_data.txt &