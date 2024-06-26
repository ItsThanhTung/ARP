# Author: Yuru Jia
# Last Modified: 2023-10-19

import random
import json
import os.path as osp

import numpy as np
from PIL import Image
import os

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from controlnet.tools.training_classes import get_class_stacks, make_one_hot, get_label_stats, get_rcs_class_probs, map_label2RGB

class GTADataset(Dataset):
    def __init__(self, args, tokenizer):
        super(GTADataset, self).__init__()

        self.file_path = args.dataset_file
        self.tokenizer = tokenizer

        self.conditioning_img_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.data = []

        with open(self.file_path) as json_data:
            self.data = json.load(json_data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        latent_path = item["latent"]
        label_file = item["semantic_path"]
        position = item["view"]
        img_type = item["type"]

        latents = torch.from_numpy(np.load(latent_path))

        # if img_type == "synthetic":
        #     if np.random.rand() < 0.1:
        #         label_file = ""

        if label_file == "" :
            label_map = np.zeros((400, 640), dtype=np.uint8)
            caption = f"A {img_type} photo taken by a fisheye camera mounted on the {position} of a car."
        else:
            label_map = np.load(label_file)
            label_map = np.array(Image.fromarray(label_map).resize((640, 400), Image.Resampling.NEAREST))
            new_texts = get_class_stacks(label_map)
            caption = f"A {img_type} photo taken by a fisheye camera mounted on the {position} of a car. The scene contains {new_texts}"

        # process cropped image label into one-hot encoding
        condition_img = make_one_hot(label_map)
        condition_img = self.conditioning_img_transforms(condition_img)
    
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids[0]
    
        return dict(pixel_values=latents, 
                    conditioning_pixel_values=condition_img, 
                    input_ids=input_ids)

class TestDataset(Dataset):
    def __init__(self, args, tokenizer):
        super(TestDataset, self).__init__()

        self.file_path = args.dataset_file
        self.tokenizer = tokenizer

        self.conditioning_img_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        with open(self.file_path) as json_data:
            self.data = json.load(json_data)


        mask_dict = {"rear" : ["/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask0.png", 
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img0.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_3.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_rear.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/samsung/roi_mask_fdd_rear.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05278_RV.png"],

                           "front" : ["/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask1.png", 
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img3.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_1.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_front.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/samsung/roi_mask_fdd_front.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05279_FV.png"],

                           "left" : ["/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask2.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img1.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_0.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_left.png",
                                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05280_MVL.png"],

                           "right" : ["/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask3.png",
                                     "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img2.png",
                                     "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_2.png",
                                      "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_right.png",
                                      "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05277_MVR.png"]}

        self.real_mask = {}

        for pos in mask_dict.keys():
            mask_paths = mask_dict[pos]
            masks = []
            for mask_path in mask_paths:
                mask = Image.open(mask_path).resize((640, 400), Image.Resampling.NEAREST)
                mask = np.array(mask)[:, :, 0] / 255.0
                masks.append(mask)

            self.real_mask[pos] = masks


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_file = item["semantic_path"]
        position = item["view"]

        masks = self.real_mask[position]
        mask = random.choice(masks)         

        label_map = np.load(label_file)
        label_map = np.array(Image.fromarray(label_map).resize((640, 400), Image.Resampling.NEAREST))
        # label_map = (label_map * mask).astype(np.uint8)

        label_image = torch.tensor(map_label2RGB(label_map).astype(np.uint8)).permute(2, 0, 1)
        new_texts = get_class_stacks(label_map)

        caption = f"A real photo taken by a fisheye camera mounted on the {position} of a car. The scene contains {new_texts}"
        # get label statistics for cropped image
        label_stats = get_label_stats(label_map)

        # process cropped image label into one-hot encoding
        condition_img = make_one_hot(label_map)
        condition_img = self.conditioning_img_transforms(condition_img)
    

        return dict(conditioning_pixel_values=condition_img, 
                    prompts=caption,
                    label_images=label_image, idx=idx)