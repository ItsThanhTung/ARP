from cleanfid import fid
import json
import shutil
import os
from tqdm import tqdm

# with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/train_fix_latent.json") as json_data:
#     all_data = json.load(json_data)
# temp_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/temp"

# os.makedirs(temp_dir, exist_ok=True)

# for data in tqdm(all_data):
#     image_path = data["img_path"]
#     new_path = os.path.join(temp_dir, os.path.basename(image_path))
#     shutil.copy(image_path, new_path)

print("start calculating")
score = fid.compute_fid("/lustre/scratch/client/vinai/users/tungdt33/ARP/sampling_data/test_24k", dataset_name="24K_REAL",
          mode="clean", dataset_split="custom")
print(f"FID: {score}")
# fid.make_custom_stats("24K_REAL", "/lustre/scratch/client/vinai/users/tungdt33/ARP/temp", mode="clean")
# 36
# 38

