# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import torch
from datetime import datetime
import os
import json
from tqdm.auto import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def load_blip2_model(pretrained_model_name_or_path: str = "Salesforce/blip2-opt-6.7b", device: str = "cuda"): # blip2-opt-6.7b
    processor = Blip2Processor.from_pretrained(pretrained_model_name_or_path)
    model = Blip2ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    model = model.to(device)

    return model, processor


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--input_path",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    args = parser.parse_args()
    return args



@torch.no_grad()
def generate_blip2_captions(
    dataset_path,
    output_file: str = "co3d_blip2_captions.json",
    prompt_per_image: int = 4,
    pretrained_model_name_or_path: str = "Salesforce/blip2-opt-6.7b",
    device: str = "cuda",
):
    # load blip2 model
    model, processor = load_blip2_model(pretrained_model_name_or_path, device)

    with open(dataset_path) as json_data:
        all_data = json.load(json_data)

    # loop over data
    new_data = []
    for idx, data in enumerate(tqdm(all_data, desc="Generate Captions")):
        # get sequence, category from batch
        # get image from batch
        img_path = data["img_path"]  # (batch_size, K=1, C, H, W)
        raw_image = [Image.open(img_path)]  # processor expects PIL images

        # run captioning
        question = "Please describe the weather conditions depicted in the image. Answer:"
        inputs = processor(raw_image, question, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        weather_prompt = processor.decode(out[0], skip_special_tokens=True).strip()

        inputs = processor(raw_image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        prompt = processor.decode(out[0], skip_special_tokens=True).strip()

        raw_image[0].save("test.png")

        new_data = data
        prompt_data = {"weather" : weather_prompt, "prompt" : prompt}
        print(prompt_data)
        new_data.update(prompt_data)
    
    with open(output_file, 'w') as f:
        json.dump(new_data, f)


if __name__ == "__main__":
    args = parse_args()
    generate_blip2_captions(args.input_path, args.output_path)