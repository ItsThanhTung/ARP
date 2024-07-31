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
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.data = []

        with open(self.file_path) as json_data:
            self.data = json.load(json_data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_file = item["img_path"]
        rgb_image = Image.open(img_file)
        if not rgb_image.mode == "RGB":
            rgb_image = rgb_image.convert("RGB")
        rgb_image = rgb_image.resize((512, 512), Image.Resampling.LANCZOS)
        instance_images = self.image_transforms(rgb_image) # 480 640

        # latent_path = item["latent"]
        label_file = item["seg_path"]
        position = item["view"]
        # img_type = item["type"]
        mask_path = item["mask"]

        mask_img = Image.open(mask_path).resize((512, 512), Image.Resampling.NEAREST)
        mask_img = (np.array(mask_img)[:, :, 0]).astype(np.uint8)

        # latents = torch.from_numpy(np.load(latent_path))

        label_map = np.load(label_file)
        label_map = np.array(Image.fromarray(label_map).resize((512, 512), Image.Resampling.NEAREST))

        label_map = np.where(mask_img == 0, 7, label_map)


        new_texts = get_class_stacks(label_map)
        caption = f"A photo taken by a fisheye camera mounted on the {position} of a car. The scene contains {new_texts}"

        # process cropped image label into one-hot encoding
        condition_img = make_one_hot(label_map)
        condition_img = self.conditioning_img_transforms(condition_img)
    
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids[0]
    
        return dict(pixel_values=instance_images, 
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

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
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
        label_file = item["seg_path"]
        position = item["view"]
        mask_path = item["mask"]

        img_file = item["img_path"]
        rgb_image = Image.open(img_file)
        if not rgb_image.mode == "RGB":
            rgb_image = rgb_image.convert("RGB")
        rgb_image = rgb_image.resize((512, 512), Image.Resampling.LANCZOS)
        instance_images = self.image_transforms(rgb_image) # 480 640

        mask_img = Image.open(mask_path).resize((512, 512), Image.Resampling.NEAREST)
        mask_img = (np.array(mask_img)[:, :, 0]).astype(np.uint8)

        mask_tensor = torch.from_numpy(mask_img).unsqueeze(-1)
        # masks = self.real_mask[position]
        # mask = random.choice(masks)         

        label_map = np.load(label_file)
        label_map = np.array(Image.fromarray(label_map).resize((512, 512), Image.Resampling.NEAREST))
        label_map = np.where(mask_img == 0, 7, label_map)
        # label_map = (label_map * mask).astype(np.uint8)

        label_image = torch.tensor(map_label2RGB(label_map).astype(np.uint8)).permute(2, 0, 1)
        new_texts = get_class_stacks(label_map)

        caption = f"A 4K photo taken by a fisheye camera mounted on the {position} of a car in snowy weather. The scene contains {new_texts}"
        # get label statistics for cropped image
        label_stats = get_label_stats(label_map)

        # process cropped image label into one-hot encoding
        condition_img = make_one_hot(label_map)
        condition_img = self.conditioning_img_transforms(condition_img)
    

        # return dict(conditioning_pixel_values=condition_img, 
        #             prompts=caption,
        #             label_images=label_image, idx=idx)

        return dict(pixel_values=instance_images,
                    conditioning_pixel_values=condition_img, 
                    prompts=caption,
                    masks=mask_tensor,
                    label_files=label_file,
                    label_images=label_image, idx=idx)