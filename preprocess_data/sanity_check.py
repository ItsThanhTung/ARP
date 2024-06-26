import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm 


JSON_PATH  = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_semantic_data.json"
INHOUSE_PALETTE = np.array([
                            [  0,   0,   0],        # unlabeled     =   0u
                            [0, 0, 142],           # road          =   1u
                            [142, 0, 0],         # obstacle      =   2u
                            [142, 142, 0],         # car mask      =   2u
                            ])

with open(JSON_PATH) as json_data:
  all_data = json.load(json_data)


import random
all_data = sorted(all_data, key=lambda x: random.random())

for data in tqdm(all_data):
    image_path = data["img_path"]
    
    mask_path = data["mask"]

    image = Image.open(image_path)
    mask = Image.open(mask_path)
    

    image = (np.array(image)[:, :, :3] / 255.0) * 2 -1
    mask = np.array(mask) / 255.0

    masked_image = np.array(((image * mask) * 0.5 + 0.5) * 255.0, dtype=np.uint8)

    seg_path = data["semantic_path"]
    if seg_path != "":
      label = np.load(seg_path)
      
      label = np.where(mask[:, :, 0], label, 3)
      label_image = INHOUSE_PALETTE[label].astype(np.uint8)
      masked_image = (masked_image * 0.5 + label_image * 0.5).astype(np.uint8)

    test_image = Image.fromarray(masked_image)
    test_image.save("/lustre/scratch/client/vinai/users/tungdt33/ARP/test.png")
    breakpoint()