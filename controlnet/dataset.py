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

class FisheyeDataset(Dataset):
    def __init__(self, args, tokenizer):
        super(FisheyeDataset, self).__init__()

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
        mask_path = item["mask"]

        mask_img = Image.open(mask_path).resize((512, 512), Image.Resampling.NEAREST)
        mask_img = (np.array(mask_img)[:, :, 0]).astype(np.uint8)

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

        self.weather = {"sunny" : "",
                        "snowy" : "in snowy weather",
                        "night" : "at night",
                        "foggy" : "in foggy weather"}

        self.weather_prompt = self.weather[args.weather_type]

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

        label_map = np.load(label_file)
        label_map = np.array(Image.fromarray(label_map).resize((512, 512), Image.Resampling.NEAREST))
        label_map = np.where(mask_img == 0, 7, label_map)

        label_image = torch.tensor(map_label2RGB(label_map).astype(np.uint8)).permute(2, 0, 1)
        new_texts = get_class_stacks(label_map)

        caption = f"A 4K photo taken by a fisheye camera mounted on the {position} of a car {self.weather_prompt}. The scene contains {new_texts}"
        # get label statistics for cropped image
        label_stats = get_label_stats(label_map)

        # process cropped image label into one-hot encoding
        condition_img = make_one_hot(label_map)
        condition_img = self.conditioning_img_transforms(condition_img)

        return dict(pixel_values=instance_images,
                    conditioning_pixel_values=condition_img, 
                    prompts=caption,
                    masks=mask_tensor,
                    label_files=label_file,
                    label_images=label_image, idx=idx)