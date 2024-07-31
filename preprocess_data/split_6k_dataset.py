import os 
import numpy as np 
from PIL import Image
import json
from collections import defaultdict

image_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/images"
segment_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/segments"

RGB_PALETTE = np.array([
    [ 0,   0,   0 ], # 0 - bg
    [128,  64, 128], # 1 - road
    [244,  35, 232], # 2 - sidewalk
    [152, 251, 152], # 3 - terrain
    [220, 20,  60 ], # 4 - person
    [ 0,   0, 142 ], # 5 - car
    [ 0,   0, 230 ], # 6 - motobike
])


MASK_DICT = {"2" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/masks/roi_mask0.png",
             "1" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/masks/roi_mask1.png",
             "0" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/masks/roi_mask2.png",
             "3" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/masks/roi_mask3.png"}
             
MASK_POS_DICT = {"2" : "left",
                 "1" : "front",
                 "0" : "rear",
                 "3" : "right",
                    }

SEQUENCE = defaultdict(list)

for image_name in os.listdir(image_dir):
    sequence_name = image_name.split("_")[0]

    img_path = os.path.join(image_dir, image_name)
    seg_path = os.path.join(segment_dir, image_name.replace("jpg", "npy"))

    assert os.path.isfile(img_path) and os.path.isfile(seg_path)
    mask_idx = image_name.split("_")[2]

    data = {"id" : "null",
            "tag" : "sim2real_inhouse_6cls",
            "img_path" : img_path,
            "seg_path": seg_path,
            "width": 1280,
            "height": 800,
            "mask": MASK_DICT[mask_idx],
            "view": MASK_POS_DICT[mask_idx],
            "num_classes": 6}

    # print(data["view"])
    # image = Image.open(data["img_path"])
    # mask = Image.open(data["mask"])
    # segment = np.load(data["seg_path"])
    # segment_image = RGB_PALETTE[segment]

    # masked_image = np.array(image)[:, :, :3] * 0.5 + np.array(segment_image) * 0.5
    # masked_image = masked_image.astype(np.uint8)
    # Image.fromarray(masked_image).save("/lustre/scratch/client/vinai/users/tungdt33/ARP/test.png")
    # breakpoint()
    SEQUENCE[sequence_name].append(data)

train_data = []
val_data = []
for idx, key in enumerate(SEQUENCE.keys()):
    if idx < 5:
        train_data += SEQUENCE[key]
    else:
        val_data += SEQUENCE[key]

print("Train: ", len(train_data))
print("Validation: ", len(val_data))

with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/train_6k.json", '+w') as f:
    json.dump(train_data, f)

with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/val_6k.json", '+w') as f:
    json.dump(val_data, f)

