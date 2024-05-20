import numpy as np 
from PIL import Image
import os
from tqdm import tqdm

TRAIN_DIR = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/MASKED_REAL/train"
TEST_DIR = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/MASKED_REAL/test"

train_txt_file = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/real_train_data.txt"
test_txt_file = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/real_test_data.txt"

rf = open(train_txt_file, "+r")
for idx, line in tqdm(enumerate(rf)):
    image_path, mask_path = line.strip().split(",")[0:2]
    image_name = "_".join(image_path.split("/")[-3:])

    try:
        image = (np.array(Image.open(image_path))[:, :, :3] / 255.0) * 2 -1
        mask = np.array(Image.open(mask_path)) / 255.0
    except Exception as error: 
        print(error)

    saved_image = np.array(((image * mask) * 0.5 + 0.5) * 255.0, dtype=np.uint8)
    Image.fromarray(saved_image).save(os.path.join(TRAIN_DIR, image_name))

rf = open(test_txt_file, "+r")
for idx, line in tqdm(enumerate(rf)):
    image_path, mask_path = line.strip().split(",")[0:2]
    image_name = "_".join(image_path.split("/")[-3:])

    try:
        image = (np.array(Image.open(image_path))[:, :, :3] / 255.0) * 2 -1
        mask = np.array(Image.open(mask_path)) / 255.0
    except Exception as error: 
        print(error)
        
    saved_image = np.array(((image * mask) * 0.5 + 0.5) * 255.0, dtype=np.uint8)
    Image.fromarray(saved_image).save(os.path.join(TEST_DIR, image_name))