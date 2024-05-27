import os 
import numpy as np 
from PIL import Image

from controlnet.tools.training_classes import (
                                                map_label2RGB
                                                )

txt_file = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/real_test_semantic_data.txt"

with open(txt_file, 'r') as f:
        # Initialize empty lists for different keys
        validation_images = []
        for line in f:
            rgb_path, label_path = line.strip().split(",")[0:2]
            validation_images.append({"image" : rgb_path, "conditioning_image" :  label_path})

for data in validation_images:
    img_file = data["image"]
    label_file = data["conditioning_image"]
    
    rgb_image = np.array(Image.open(img_file).resize((640, 480), Image.Resampling.LANCZOS))
    label_map = np.load(label_file)
    label_map = np.array(Image.fromarray(label_map).resize((640, 480), Image.Resampling.NEAREST))
    label_image = np.array(Image.fromarray(map_label2RGB(label_map).astype(np.uint8)))
    

    test_image = Image.fromarray((rgb_image * 0.5 + label_image * 0.5).astype(np.uint8))
    test_image.save("/lustre/scratch/client/vinai/users/tungdt33/ARP/test.png")
    print(img_file, label_file)
    breakpoint()