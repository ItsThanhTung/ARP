import os 
import numpy as np 
from PIL import Image
from tqdm import tqdm
import json
import random


# label_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/semantic_annotations/gtLabels"
# save_folder = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/semantic_annotations/processed_6_classes"
# # RGB_PALETTE = np.array([
# #     [ 0,   0,   0 ], # 0 - bg
# #     [128,  64, 128], # 1 - road
# #     [244,  35, 232], # 2 - sidewalk
# #     [152, 251, 152], # 3 - terrain
# #     [220, 20,  60 ], # 4 - person
# #     [ 0,   0, 142 ], # 5 - car
# #     [ 0,   0, 230 ], # 6 - motobike
# # ])

# # # "class_names": [
# # #        0 "void", -> 0
# # #        1 "road", -> 1
# # #        2 "lanemarks", -> 1
# # #        3 "curb", -> 2
# # #        4 "person", -> 4
# # #        5 "rider", -> 4
# # #        6 "vehicles", -> 5
# # #        7 "bicycle", -> 6
# # #        8 "motorcycle", -> 6
# # #        9 "traffic_sign"
# # #     ],              

# for label_name in tqdm(os.listdir(label_dir)):
#     label_img = np.array(Image.open(os.path.join(label_dir, label_name)))
#     label_img = np.where(label_img == 2, 0, label_img)
#     label_img = np.where(label_img == 3, 2, label_img)
#     label_img = np.where(label_img == 5, 4, label_img)
#     label_img = np.where(label_img == 6, 5, label_img)
#     label_img = np.where(label_img == 7, 6, label_img)
#     label_img = np.where(label_img == 8, 6, label_img)
#     label_img = np.where(label_img == 9, 0, label_img)
    
#     np.save(os.path.join(save_folder, label_name + '.npy'), label_img)


image_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/rgb_alltraintest"
label_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/semantic_annotations/processed_6_classes"

all_label = sorted(os.listdir(label_dir))
train_label_dir = all_label[:5000]
val_label_dir = all_label[5000:]


val_data = []
for idx, label_name in tqdm(enumerate(val_label_dir)):
    label_path = os.path.join(label_dir, label_name)
    image_path = os.path.join(image_dir, label_name[:-4])
    assert os.path.isfile(image_path)

    data = {"id" : 0, 
            "tag" : "woodscape",
            "img_path" : image_path,
            "seg_path" : label_path,
            "width" : 1280,
            "height" : 966}
    val_data.append(data)

with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/val.json", '+w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

val_data_1k = random.sample(val_data, 1000)
with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/val_1k.json", '+w', encoding='utf-8') as f:
    json.dump(val_data_1k, f, ensure_ascii=False, indent=4)

train_data = []
for idx, label_name in tqdm(enumerate(train_label_dir)):
    label_path = os.path.join(label_dir, label_name)
    image_path = os.path.join(image_dir, label_name[:-4])
    assert os.path.isfile(image_path)

    data = {"id" : 0, 
            "tag" : "woodscape",
            "img_path" : image_path,
            "seg_path" : label_path,
            "width" : 1280,
            "height" : 966}
    train_data.append(data)

train_data_500 = random.sample(train_data, 500)
with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/train_500.json", '+w', encoding='utf-8') as f:
    json.dump(train_data_500, f, ensure_ascii=False, indent=4)

train_data_1k = random.sample(train_data, 1000)
with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/train_1k.json", '+w', encoding='utf-8') as f:
    json.dump(train_data_1k, f, ensure_ascii=False, indent=4)

train_data_2k = random.sample(train_data, 2000)
with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/train_2k.json", '+w', encoding='utf-8') as f:
    json.dump(train_data_2k, f, ensure_ascii=False, indent=4)

train_data_3k = random.sample(train_data, 3000)
with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/woodscape/train_3k.json", '+w', encoding='utf-8') as f:
    json.dump(train_data_3k, f, ensure_ascii=False, indent=4)

