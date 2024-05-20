import numpy as np 
from PIL import Image
import glob
import os
from os import path as osp
from os.path import join as ospj

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
TRAIN_TXT_PATH=ospj(OUT_TXT_DIR, "real_train_data.txt")
TEST_TXT_PATH=ospj(OUT_TXT_DIR, "real_test_data.txt")

ALL_DATA = []
TRAIN_DATA = []
TEST_DATA = []

EXCLUDED_SUBSET= ["2022-08-10-15-18-13",
                  "2022-08-10-15-19-15",
                  "2022-08-10-15-22-18",
                  "2022-08-10-15-23-19",
                  "2022-08-10-15-24-20",
                  "2022-08-10-18-05-03",
                  "2022-08-10-18-08-46",
                  "2022-08-10-18-11-35",
                  "2022-08-10-18-15-19"]

DATE_DIR="/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2022-08-10"
MASK_DICT = {"0" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask0.png",
             "1" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask1.png",
             "2" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask2.png",
             "3" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vfe34/roi_mask3.png"}

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
        str_data = image_path + "," + MASK_DICT[mask_idx] 

        if subset in EXCLUDED_SUBSET:
            TRAIN_DATA.append(str_data)
        else:
            ALL_DATA.append(str_data)


DATE_DIR="/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2023-06-02"
MASK_DICT = {"im0" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img0.png",
             "im1" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img1.png",
             "im2" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img2.png",
             "im3" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/togg/roi_mask_togg_img3.png"}

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
        str_data = image_path + "," + MASK_DICT[mask_idx]

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
        str_data = image_path + "," + MASK_DICT[mask_idx]

        ALL_DATA.append(str_data)



DATE_DIR= ["/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2024-01-15",
           "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2024-01-22",
           "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/inhouse/segment/rgb/2024-03-24"]

MASK_DICT = {"front" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_front.png",
             "rear" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_rear.png",
             "left" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_left.png",
             "right" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/vf8/roi_mask_vf8_img_right.png"}

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
            str_data = image_path + "," + MASK_DICT[mask_idx]

            ALL_DATA.append(str_data)



data_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/samsung_fdd/images"
MASK_DICT = {"front" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/samsung/roi_mask_fdd_front.png",
             "rear" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/samsung/roi_mask_fdd_rear.png"}

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
    str_data = image_path + "," + MASK_DICT[mask_idx]

    TRAIN_DATA.append(str_data)


data_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/rgb_alltraintest"
MASK_DICT = {"FV" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05279_FV.png",
             "RV" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05278_RV.png", 
             "MVR" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05277_MVR.png", 
             "MVL" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/woodscape/roi_mask_woodscape_05280_MVL.png"}

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
    str_data = image_path + "," + MASK_DICT[mask_idx]

    TRAIN_DATA.append(str_data)



from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
train_data, test_data = train_test_split(ALL_DATA, test_size=0.2, random_state=42)

# Print the sizes of the splits to verify
print(f"Training data size: {len(train_data)}")
print(f"Testing data size: {len(test_data)}")

TRAIN_DATA += train_data
TEST_DATA = test_data

print(f"ALL training data size: {len(TRAIN_DATA)}")
print(f"ALL testing data size: {len(TEST_DATA)}")

wf = open(TRAIN_TXT_PATH, "+w")
for data in TRAIN_DATA:
    wf.write(data + "\n")

wf.close()

wf = open(TEST_TXT_PATH, "+w")
for data in TEST_DATA:
    wf.write(data + "\n")
wf.close()