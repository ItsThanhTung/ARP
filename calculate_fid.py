from cleanfid import fid
import json
import shutil
import os
from tqdm import tqdm
import tyro

def calculate_fid(data_root: str):
    print("start calculating")
    score = fid.compute_fid(data_root, dataset_name="24K_REAL", mode="clean", dataset_split="custom")
    print(score)

if __name__ == "__main__":
    tyro.cli(calculate_fid)


