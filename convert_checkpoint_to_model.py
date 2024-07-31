# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import json
import tyro
import shutil

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)



def convert_checkpoint_to_model(
    checkpoint_path: str, output_dir: str, pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
):
    unet = UNet2DConditionModel.from_pretrained(checkpoint_path)

    pipeline = StableDiffusionPipeline.from_pretrained(
                                                            pretrained_model_name_or_path,
                                                            unet=unet,
                                                        )
                                                        
    pipeline.save_pretrained(output_dir)


if __name__ == "__main__":
    tyro.cli(convert_checkpoint_to_model)