import os 
import json
from PIL import Image
import numpy as np
import tyro
from tqdm import tqdm

INHOUSE_PALETTE = np.array([
        [  0,   0,   0],  # unlabeled     =   0u
        [0, 142, 142],     # road          =   1u
        [142, 0, 0],  # obstacle      =   2u

])

def generate_json(data_root: str, output_json_path: str):
    json_data = []
    image_dir = os.path.join(data_root, "images")
    label_dir = os.path.join(data_root, "labels")

    tag = os.path.basename(data_root)
    all_image_dirs = os.listdir(image_dir)

    for idx, image_name in tqdm(enumerate(all_image_dirs)):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace("png", "npy"))

        assert os.path.isfile(image_path) and os.path.isfile(label_path) 

        image = np.array(Image.open(image_path))
        semantic_label = np.load(label_path)
        semantic_label = np.array(Image.fromarray(semantic_label).resize((512, 512), Image.Resampling.NEAREST))
        np.save(label_path, semantic_label)     

        # label_img = INHOUSE_PALETTE[semantic_label]
        # masked_image =np.array(image * 0.5 + label_img * 0.5, dtype=np.uint8)
        # img = Image.fromarray(masked_image)
        # img.save("/lustre/scratch/client/vinai/users/tungdt33/ARP/test.png")
        # breakpoint()
        data = {"id" :idx, 
                "tag" : tag,
                "img_path" : image_path,
                "seg_path" : label_path,
                "width" : 512,
                "height" : 512}
        json_data.append(data)

    with open(output_json_path, '+w') as f:
        json.dump(json_data, f)

if __name__ == "__main__":
    tyro.cli(generate_json)
    