import os
import numpy as np 
from PIL import Image
from tqdm import tqdm
import json


image_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb"
annotation_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/annotation/cvat_processed_3_classes"


JSON_PATH = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_latent_data.json"
NEW_JSON_PATH  = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_semantic_latent_data.json"

TRAIN_DATA_DICT = {}
with open(JSON_PATH) as json_data:
    TRAIN_DATA = json.load(json_data)


for data in TRAIN_DATA:
    rgb_path = data["img_path"] 
    TRAIN_DATA_DICT[rgb_path] = {"semantic_path" : ""}


mapping_folder = {"2022-08-10-15-18-13" : "2022-08-10/2022-08-10-15-18-13",
                "2022-08-10-15-19-15" : "2022-08-10/2022-08-10-15-19-15",
                "2022-08-10-15-23-19" : "2022-08-10/2022-08-10-15-23-19",
                "2022-08-10-15-24-20" : "2022-08-10/2022-08-10-15-24-20",
                "2022-08-10-18-05-03" : "2022-08-10/2022-08-10-18-05-03",
                "2022-08-10-18-11-35" : "2022-08-10/2022-08-10-18-11-35",
                "2022-08-10-18-15-19" : "2022-08-10/2022-08-10-18-15-19",
                "2022-12-14-16-43-38" : "2023-06-06/2022-12-14-16-43-38",
                "2022-12-14-16-48-57" : "2023-06-06/2022-12-14-16-48-57",

                "2023-06-06-0003" : "2023-06-06/2023-06-06-0003",
                "1685604028864" : "2023-06-02/1685604028864",

                "vinai_ddsbag_3.0.3_2023-08-28_02-19-34" : "2024-03-24/vinai_ddsbag_3.0.3_2023-08-28_02-19-34",
                "vinai_ddsbag_3.0.3_2023-08-28_05-03-52" : "2024-03-24/vinai_ddsbag_3.0.3_2023-08-28_05-03-52",
                "vinai_ddsbag_3.0.3_2023-09-02_21-32-41" : "2024-03-24/vinai_ddsbag_3.0.3_2023-09-02_21-32-41",

                "vinai_ddsbag_1701_Testing_Basement_Subset" : "2024-01-15/vinai_ddsbag_1701_Testing_Basement_Subset",
                "vinai_ddsbag_2023-08-18_00-47-15_Sunflare" : "2024-01-15/vinai_ddsbag_2023-08-18_00-47-15_Sunflare",

                "vinai_ddsbag_2023-08-20_13-11-48_20240116_Data" : "2024-01-22/vinai_ddsbag_2023-08-20_13-11-48_20240116_Data",
                "vinai_ddsbag_2023-08-20_21-14-28_20240118_Data" : "2024-01-22/vinai_ddsbag_2023-08-20_21-14-28_20240118_Data",
                "vinai_ddsbag_2023-08-20_21-21-41_20240118_Data" : "2024-01-22/vinai_ddsbag_2023-08-20_21-21-41_20240118_Data",
                "vinai_ddsbag_2023-10-18_14-53-41_forward_straight_rainy" : "2024-01-22/vinai_ddsbag_2023-10-18_14-53-41_forward_straight_rainy",
                "vinai_ddsbag_2023-10-19_02-02-40_snow" : "2024-01-22/vinai_ddsbag_2023-10-19_02-02-40_snow",

                "vinai_ddsbag_OGM_Sidewalk_Green_Grass" : "2024-01-15/vinai_ddsbag_OGM_Sidewalk_Green_Grass"}

count = 0
for anno_name, image_name in mapping_folder.items():
    image_folder = os.path.join(image_dir, image_name)
    annotation_folder = os.path.join(annotation_dir, anno_name)
    

    for image_file in os.listdir(image_folder):
        if image_file.split(".")[-1] not in ["jpg", "png"]:
            continue
        image_path = os.path.join(image_folder, image_file)
        annotation_path = os.path.join(annotation_folder, image_file.split(".")[0] + ".npy")

        if not os.path.isfile(annotation_path):
            continue

        assert os.path.isfile(image_path), f"something was wrong at, cant find {image_path}"

        assert image_path in TRAIN_DATA_DICT
        TRAIN_DATA_DICT[image_path] = {"semantic_path" : annotation_path}
        count += 1
      

mapping_folder = {"/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/semantic_annotations/processed_3_classes/2022-29-11-8234" : \
                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/rgb_alltraintest",
                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/samsung_fdd/cvat_processed_3_classes/2023-16-03-3897" :\
                    "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/samsung_fdd/images"}

for annotation_folder, image_folder in mapping_folder.items():

    for image_file in os.listdir(image_folder):
        if image_file.split(".")[-1] not in ["jpg", "png"]:
            continue
        image_path = os.path.join(image_folder, image_file)
        annotation_path = os.path.join(annotation_folder, image_file + ".npy")

        if not os.path.isfile(annotation_path):
            continue

        assert os.path.isfile(image_path), f"something was wrong at, cant find {image_path}"

        assert image_path in TRAIN_DATA_DICT
        TRAIN_DATA_DICT[image_path] = {"semantic_path" : annotation_path}
        count += 1

print(f"Num semantic file: {count}")


NEW_TRAIN_DATA = []
for data in TRAIN_DATA:
    rgb_path = data["img_path"]
    data.update(TRAIN_DATA_DICT[rgb_path])
    NEW_TRAIN_DATA.append(data)


with open(NEW_JSON_PATH, '+w') as f:
    json.dump(NEW_TRAIN_DATA, f)
