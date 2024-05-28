# Author: Yuru Jia
# Last Modified: 2023-10-19

import random
import json
import os.path as osp

import numpy as np
from PIL import Image


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

        with open(self.file_path, 'rt') as f:
            for line in f:
                rgb_path, label_path = line.strip().split(",")[0:2]
                self.data.append({"image" : rgb_path, "conditioning_image" :  label_path})
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_file = item["image"]
        label_file = item["conditioning_image"]

        latent_path = img_file.replace("train", "train_latent").split(".")[0] + ".npy"
        latents = torch.from_numpy(np.load(latent_path))

        label_map = np.load(label_file)
        label_map = np.array(Image.fromarray(label_map).resize((640, 480), Image.Resampling.NEAREST))
        new_texts = get_class_stacks(label_map)

        caption = f"A fisheye image contain {new_texts}"
        # get label statistics for cropped image
        label_stats = get_label_stats(label_map)

        # process cropped image label into one-hot encoding
        condition_img = make_one_hot(label_map)
        condition_img = self.conditioning_img_transforms(condition_img)
    
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids[0]


        return dict(pixel_values=latents, 
                    conditioning_pixel_values=condition_img, 
                    input_ids=input_ids,
                    label_stats=label_stats)

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

        self.data = []

        with open(self.file_path, 'rt') as f:
            for line in f:
                rgb_path, label_path = line.strip().split(",")[0:2]
                self.data.append({"image" : rgb_path, "conditioning_image" :  label_path})
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_file = item["image"]
        label_file = item["conditioning_image"]

        label_map = np.load(label_file)
        label_map = np.array(Image.fromarray(label_map).resize((640, 480), Image.Resampling.NEAREST))
        label_image = torch.tensor(map_label2RGB(label_map).astype(np.uint8)).permute(2, 0, 1)
        new_texts = get_class_stacks(label_map)

        caption = f"A fisheye image contain {new_texts}"
        # get label statistics for cropped image
        label_stats = get_label_stats(label_map)

        # process cropped image label into one-hot encoding
        condition_img = make_one_hot(label_map)
        condition_img = self.conditioning_img_transforms(condition_img)
    

        return dict(conditioning_pixel_values=condition_img, 
                    prompts=caption,
                    label_images=label_image, idx=idx)