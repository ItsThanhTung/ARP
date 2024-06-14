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
import json
from tqdm import tqdm



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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    args = parser.parse_args()
    return args



def main(args):
    new_data = []
    with open(args.input_path) as json_data:
        all_data = json.load(json_data)

    for data in tqdm(all_data):
        image_path = str(data["id"]) + "_"  + data["tag"] + "_" + os.path.basename(data["img_path"])
        save_path = image_path.split(".")[0] + ".npy"
        save_path = os.path.join(args.output_dir, save_path)

        if os.path.isfile(save_path):
            data["latent"] = save_path
            new_data.append(data)
        else:
            print(f"Missing latent file at {save_path}")
            data["latent"] = "save_path"
            new_data.append(data)

    with open(args.output_path, '+w') as f:
        json.dump(new_data, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)