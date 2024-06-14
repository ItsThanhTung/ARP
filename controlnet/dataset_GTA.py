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
        
        pos_dict = {"image_1" : "front", "image_3" : "right", "image_2" : "left" }
        self.data = []
        images_dir = os.path.join(self.file_path, "images")
        segments_dir = os.path.join(self.file_path, "segments_fixed")
        for path in os.listdir(images_dir):
            image_path = os.path.join(images_dir, path)
            segment_path = os.path.join(segments_dir, path.split(".")[0] + ".npy")

            assert os.path.isfile(image_path), f"cannot find image file {image_path}"
            assert os.path.isfile(segment_path), f"cannot find segment file {segment_path}"


            for key in pos_dict.keys():
                if key in path:
                    view = pos_dict[key]
                    break

            self.data.append({"img_path" : image_path, "seg_path" : segment_path, "view" : view})

        # self.image_paths = [os.path.join(images_dir, path) for path in os.listdir(images_dir)]

        # with open(self.file_path) as json_data:
        #     self.data = json.load(json_data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        latent_path = item["img_path"]
        label_file = item["seg_path"]
        position = item["view"]
        
        instance_image = Image.open(latent_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        instance_image = instance_image.resize((640, 400), Image.BILINEAR)
        instance_image = self.image_transforms(instance_image) # 480 640

        label_map = np.load(label_file)
        label_map = np.array(Image.fromarray(label_map).resize((640, 400), Image.Resampling.NEAREST))
        new_texts = get_class_stacks(label_map)

        caption = f"A photo taken by a fisheye camera mounted on the {position} of a car. The scene contains {new_texts}"
        # get label statistics for cropped image
        label_stats = get_label_stats(label_map)

        # process cropped image label into one-hot encoding
        condition_img = make_one_hot(label_map)
        condition_img = self.conditioning_img_transforms(condition_img)
    
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids[0]


        return dict(pixel_values=instance_image, 
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
                rgb_path, label_path, position = line.strip().split(",")[0:3]
                self.data.append({"image" : rgb_path, "conditioning_image" :  label_path, "position" : position})
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_file = item["image"]
        label_file = item["conditioning_image"]
        position = item["position"]

        label_map = np.load(label_file)
        label_map = np.array(Image.fromarray(label_map).resize((640, 400), Image.Resampling.NEAREST))
        label_image = torch.tensor(map_label2RGB(label_map).astype(np.uint8)).permute(2, 0, 1)
        new_texts = get_class_stacks(label_map)

        caption = f"A photo taken by a fisheye camera mounted on the {position} of a car. The scene contains {new_texts}"
        # get label statistics for cropped image
        label_stats = get_label_stats(label_map)

        # process cropped image label into one-hot encoding
        condition_img = make_one_hot(label_map)
        condition_img = self.conditioning_img_transforms(condition_img)
    

        return dict(conditioning_pixel_values=condition_img, 
                    prompts=caption,
                    label_images=label_image, idx=idx)