import os
import numpy as np 
from PIL import Image
from tqdm import tqdm

image_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb"
annotation_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/annotation/cvat_processed_3_classes"

TRAIN_DIR = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/MASKED_REAL/train"
TEST_DIR = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/MASKED_REAL/test"

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


train_paths = []
test_paths = []

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

        masked_image_name = "_".join(image_path.split("/")[-3:])

        data = {"label" : annotation_path}
        
        if os.path.isfile(os.path.join(TRAIN_DIR, masked_image_name)):
            masked_image_path = os.path.join(TRAIN_DIR, masked_image_name)
            data["rgb"] = masked_image_path
            train_paths.append(data)

        elif os.path.isfile(os.path.join(TEST_DIR, masked_image_name)):
            os.path.isfile(os.path.join(TEST_DIR, masked_image_name))
            data["rgb"] = masked_image_path
            test_paths.append(data)

        else:
            raise Exception("should not be here")


print(f"ALL training data size: {len(train_paths)}")
print(f"ALL testing data size: {len(test_paths)}")

OUT_TXT_DIR="/lustre/scratch/client/vinai/users/tungdt33/ARP/data"
TRAIN_TXT_PATH=os.path.join(OUT_TXT_DIR, "real_train_semantic_data.txt")
TEST_TXT_PATH=os.path.join(OUT_TXT_DIR, "real_test_semantic_data.txt")

wf = open(TRAIN_TXT_PATH, "+w")
for data in train_paths:
    wf.write(data["rgb"] + "," + data["label"] + "\n")

wf.close()

wf = open(TEST_TXT_PATH, "+w")
for data in test_paths:
    wf.write(data["rgb"] + "," + data["label"] + "\n")
wf.close()