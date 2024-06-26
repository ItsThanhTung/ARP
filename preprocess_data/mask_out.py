import numpy as np 
from PIL import Image
import glob
import os
from os import path as osp
from os.path import join as ospj
import json

def test_mask(str_data):
    print(str_data)
    image_path = str_data.split(",")[0]
    mask_path = str_data.split(",")[1]

    image = np.array(Image.open(image_path))[:, :, :3]
    mask = np.array(Image.open(mask_path))
    saved_image = np.array(image * 0.5 + mask * 0.5, dtype=np.uint8)
    Image.fromarray(saved_image).save("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/test_mask.png")
    breakpoint()

TOTAL_IMAGES=0
OUT_TXT_DIR="/lustre/scratch/client/vinai/users/tungdt33/ARP/data"
TRAIN_TXT_PATH=ospj(OUT_TXT_DIR, "real_train_data.json")
TEST_TXT_PATH=ospj(OUT_TXT_DIR, "real_test_data.json")

ALL_DATA = []
TRAIN_DATA = []
TEST_DATA = []


DATE_DIR="/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2022-08-10"
MASK_DICT = {"0" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask0.png",
             "1" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask1.png",
             "2" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask2.png",
             "3" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask3.png"}
             
MASK_POS_DICT = {"0" : "rear",
                 "1" : "front",
                 "2" : "left",
                 "3" : "right",
                    }

for key, value in MASK_DICT.items():
    if not os.path.isfile(value):
        print(f"Could not find {value}")
        raise Exception

for subset in os.listdir(DATE_DIR):
    data_dir = ospj(DATE_DIR, subset)
    for image_name in os.listdir(data_dir):
        if image_name.split(".")[-1] not in ["png", "jpg"]:
            continue
        mask_idx = image_name.split("_")[1]

        if mask_idx not in MASK_DICT:
            print(f"Something is wrong at {image_name} with mask_idx={mask_idx}")
        
        image_path = ospj(data_dir, image_name)
        str_data = image_path + "," + MASK_DICT[mask_idx] + "," + MASK_POS_DICT[mask_idx] 

        str_data = {"tag" : "2022-08-10", 
                    "img_path" : image_path,
                    "mask" : MASK_DICT[mask_idx],
                    "view" : MASK_POS_DICT[mask_idx],
                   }

        ALL_DATA.append(str_data)


DATE_DIR="/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2023-06-02"
MASK_DICT = {"im0" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img0.png",
             "im1" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img1.png",
             "im2" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img2.png",
             "im3" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img3.png"}

MASK_POS_DICT = {"im0" : "rear",
                 "im1" : "left",
                 "im2" : "right",
                 "im3" : "front",
                    }

for key, value in MASK_DICT.items():
    if not os.path.isfile(value):
        print(f"Could not find {value}")
        raise Exception

for subset in os.listdir(DATE_DIR):
    data_dir = ospj(DATE_DIR, subset)
    for image_name in os.listdir(data_dir):
        if image_name.split(".")[-1] not in ["png", "jpg"]:
            continue
        mask_idx = image_name.split("_")[1]

        if mask_idx not in MASK_DICT:
            print(f"Something is wrong at {image_name} with mask_idx={mask_idx}")
        
        image_path = ospj(data_dir, image_name)
        str_data = image_path + "," + MASK_DICT[mask_idx] + "," + MASK_POS_DICT[mask_idx] 

        str_data = {"tag" : "2023-06-02", 
                    "img_path" : image_path,
                    "mask" : MASK_DICT[mask_idx],
                    "view" : MASK_POS_DICT[mask_idx],
                    }

        TRAIN_DATA.append(str_data)    # FOR TOGG dataset, we only use for training



DATE_DIR="/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2023-06-06"
MASK_DICT = {"cam0" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_0.png",
             "cam1" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_1.png",
             "cam2" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_2.png",
             "cam3" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_3.png",
             "front" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_1.png",
             "rear" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_3.png",
             "left" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_0.png",
             "right" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/ford/roi_mask_ford_2.png"}


MASK_POS_DICT = {"cam0" : "left",
                "cam1" : "front",
                "cam2" : "right",
                "cam3" : "rear",
                "front" : "front",
                "rear" : "rear",
                "left" : "left",
                "right" : "right"}

for key, value in MASK_DICT.items():
    if not os.path.isfile(value):
        print(f"Could not find {value}")
        raise Exception

for subset in os.listdir(DATE_DIR):
    data_dir = ospj(DATE_DIR, subset)
    for image_name in os.listdir(data_dir):
        if image_name.split(".")[-1] not in ["png", "jpg"] or "topview" in image_name:
            continue
        
        if "right" in image_name:
            mask_idx = "right"
        elif "left" in image_name:
            mask_idx = "left"
        elif "front" in image_name:
            mask_idx = "front"
        elif "rear" in image_name:  
            mask_idx = "rear"
        else:
            mask_idx = image_name.split("_")[0]


        if mask_idx not in MASK_DICT:
            print(f"Something is wrong at {image_name} with mask_idx={mask_idx}")
        
        image_path = ospj(data_dir, image_name)
        str_data = image_path + "," + MASK_DICT[mask_idx] + "," + MASK_POS_DICT[mask_idx] 

        str_data = {"tag" : "2023-06-06", 
                    "img_path" : image_path,
                    "mask" : MASK_DICT[mask_idx],
                    "view" : MASK_POS_DICT[mask_idx],
                    }
        ALL_DATA.append(str_data)



DATE_DIR= ["/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2024-01-15",
           "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2024-01-22",
           "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2024-03-24"]

MASK_DICT = {"front" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_front.png",
             "rear" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_rear.png",
             "left" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_left.png",
             "right" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_right.png"}

MASK_POS_DICT = {
                    "front" : "front",
                    "rear" : "rear",
                    "left" : "left",
                    "right" : "right"}


for key, value in MASK_DICT.items():
    if not os.path.isfile(value):
        print(f"Could not find {value}")
        raise Exception

for date_dir in DATE_DIR:
    for subset in os.listdir(date_dir):
        data_dir = ospj(date_dir, subset)
        for image_name in os.listdir(data_dir):
            if image_name.split(".")[-1] not in ["png", "jpg"] or "topview" in image_name:
                continue
            
            mask_idx = image_name.split("_")[1]


            if mask_idx not in MASK_DICT:
                print(f"Something is wrong at {image_name} with mask_idx={mask_idx}")
            
            image_path = ospj(data_dir, image_name)
            str_data = image_path + "," + MASK_DICT[mask_idx] + "," + MASK_POS_DICT[mask_idx] 

            str_data = {"tag" : "vf8", 
                    "img_path" : image_path,
                    "mask" : MASK_DICT[mask_idx],
                    "view" : MASK_POS_DICT[mask_idx],
                    }

            ALL_DATA.append(str_data)



data_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/samsung_fdd/images"
MASK_DICT = {"front" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/samsung/roi_mask_fdd_front.png",
             "rear" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/samsung/roi_mask_fdd_rear.png"}

MASK_POS_DICT = {
                "front" : "front",
                "rear" : "rear"}

for key, value in MASK_DICT.items():
    if not os.path.isfile(value):
        print(f"Could not find {value}")
        raise Exception

for image_name in os.listdir(data_dir):
    if image_name.split(".")[-1] not in ["png", "jpg"]:
        continue
    
    mask_idx = image_name.split("_")[1]

    if mask_idx not in MASK_DICT:
        print(f"Something is wrong at {image_name} with mask_idx={mask_idx}")
    
    image_path = ospj(data_dir, image_name)
    str_data = image_path + "," + MASK_DICT[mask_idx] + "," + MASK_POS_DICT[mask_idx] 

    str_data = {"tag" : "samsung_fdd", 
                "img_path" : image_path,
                "mask" : MASK_DICT[mask_idx],
                "view" : MASK_POS_DICT[mask_idx],
                }

    TRAIN_DATA.append(str_data)


data_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/rgb_alltraintest"
MASK_DICT = {"FV" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05279_FV.png",
             "RV" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05278_RV.png", 
             "MVR" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05277_MVR.png", 
             "MVL" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05280_MVL.png"}

MASK_POS_DICT = {
                "FV" : "front",
                "MVR" : "right",
                "MVL" : "left",
                "RV" : "rear"}

for key, value in MASK_DICT.items():
    if not os.path.isfile(value):
        print(f"Could not find {value}")
        raise Exception

for image_name in os.listdir(data_dir):
    if image_name.split(".")[-1] not in ["png", "jpg"]:
        continue
    
    mask_idx = image_name.split("_")[1].split(".")[0]

    if mask_idx not in MASK_DICT:
        print(f"Something is wrong at {image_name} with mask_idx={mask_idx}")
    
    image_path = ospj(data_dir, image_name)
    str_data = image_path + "," + MASK_DICT[mask_idx] + "," + MASK_POS_DICT[mask_idx] 

    str_data = {"tag" : "woodscape", 
                "img_path" : image_path,
                "mask" : MASK_DICT[mask_idx],
                "view" : MASK_POS_DICT[mask_idx],
                }

    TRAIN_DATA.append(str_data)



from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
train_data, test_data = train_test_split(ALL_DATA, test_size=0.2, random_state=42)

TRAIN_DATA += train_data
TEST_DATA = test_data

print(f"ALL training data size: {len(TRAIN_DATA)}")
print(f"ALL testing data size: {len(TEST_DATA)}")


with open(TRAIN_TXT_PATH, '+w') as f:
    json.dump(TRAIN_DATA, f)

with open(TEST_TXT_PATH, '+w') as f:
    json.dump(TEST_DATA, f)

