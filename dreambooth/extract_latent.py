# Obtained from: https://github.com/huggingface/diffusers/tree/main/examples/dreambooth

import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, model_info, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )

    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
    ):

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        # self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.data = []
        with open(self.instance_data_root, 'rt') as f:
            for line in f:
                rgb_path, mask_path, position = line.strip().split(",")[0:3]
                self.data.append({"image" : rgb_path, "mask" :  mask_path,
                                    "position" : position})


        self.num_instance_images = len(self.data)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
       
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        data_item = self.data[index % self.num_instance_images]
        example["image_path"] = "_".join(data_item["image"].split("/")[-3:]).split(".")[0]  + "_" + data_item["position"] + ".png"
        example["mask_path"] = data_item["mask"]

        try:
            image = (np.array(Image.open(data_item["image"]))[:, :, :3] / 255.0) * 2 -1
            mask = np.array(Image.open(data_item["mask"])) / 255.0
            instance_image = np.array(((image * mask) * 0.5 + 0.5) * 255.0, dtype=np.uint8)
            instance_image = Image.fromarray(instance_image)
        except:
            return self.__getitem__(0)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        
        instance_image = instance_image.resize((640, 400), Image.BILINEAR)
        example["instance_images"] = self.image_transforms(instance_image) # 480 640
        # Image.fromarray(((example["instance_images"] * 0.5 + 0.5).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)).save("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/test_data.png")
        # breakpoint()
        return example



def model_has_vae(args):
    config_file_name = os.path.join("vae", AutoencoderKL.config_name)
    if os.path.isdir(args.pretrained_model_name_or_path):
        config_file_name = os.path.join(args.pretrained_model_name_or_path, config_file_name)
        return os.path.isfile(config_file_name)
    else:
        files_in_repo = model_info(args.pretrained_model_name_or_path, revision=args.revision).siblings
        return any(file.rfilename == config_file_name for file in files_in_repo)


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no")

    os.makedirs(args.output_dir, exist_ok=True)

    # import correct text encoder class
    if model_has_vae(args):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    else:
        vae = None

    if vae is not None:
        vae.requires_grad_(False)

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
    )
    train_dataset[0] # for debugging

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        persistent_workers=True,
    )

    weight_dtype = torch.float32
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    train_dataloader = accelerator.prepare(train_dataloader)
    for step, batch in tqdm(enumerate(train_dataloader)):
        model_input = vae.encode(batch["instance_images"].to(dtype=weight_dtype)).latent_dist.sample()

        for image_path, precompute_latent in zip(batch["image_path"], model_input):
            save_path = os.path.basename(image_path).split(".")[0] + ".npy"
            save_path = os.path.join(args.output_dir, save_path)
            np.save(save_path, precompute_latent.cpu().numpy())
                

if __name__ == "__main__":
    args = parse_args()
    main(args)