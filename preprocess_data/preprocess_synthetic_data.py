import os 
import numpy as np 
from PIL import Image
import cv2


DATA_ROOT = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/SYNTHETIC/extracted"
OUT_ROOT = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/SYNTHETIC/MASKED_SYNTHETIC"
OUT_IMG_ROOT = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/SYNTHETIC/MASKED_SYNTHETIC/images"
OUT_SEMANTIC_ROOT = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/SYNTHETIC/MASKED_SYNTHETIC/labels"

TRAIN_TXT_PATH = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/synthetic_train_data.txt"
TEST_TXT_PATH = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/synthetic_test_data.txt"

os.makedirs(OUT_IMG_ROOT, exist_ok=True)
os.makedirs(OUT_SEMANTIC_ROOT, exist_ok=True)

list_extracted_topics = [
    'fisheye_front', 
    'fisheye_left', 
    'fisheye_rear', 
    'fisheye_right', 
]


SYN_CARLA_INHOUSE_PALETTE = np.array([
        [  0,   0,   0],  # unlabeled     =   0u
        [128,  64, 128],  # road          =   1u
        [244,  35, 232],  # sidewalk      =   2u
        [ 70,  70,  70],  # building      =   3u
        [102, 102, 156],  # wall          =   4u
        [190, 153, 153],  # fence         =   5u
        [153, 153, 153],  # pole          =   6u
        [250, 170,  30],  # traffic light =   7u
        [220, 220,   0],  # traffic sign  =   8u
        [107, 142,  35],  # vegetation    =   9u
        [152, 251, 152],  # terrain       =  10u
        [ 70, 130, 180],  # sky           =  11u
        [220,  20,  60],  # pedestrian    =  12u
        [255,   0,   0],  # rider         =  13u
        [  0,   0, 142],  # Car           =  14u
        [  0,   0,  70],  # truck         =  15u
        [  0,  60, 100],  # bus           =  16u
        [  0,  80, 100],  # train         =  17u
        [  0,   0, 230],  # motorcycle    =  18u
        [119,  11,  32],  # bicycle       =  19u
        [110, 190, 160],  # static        =  20u
        [170, 120,  50],  # dynamic       =  21u
        [ 55,  90,  80],  # other         =  22u
        [ 45,  60, 150],  # water         =  23u
        [157, 234,  50],  # road line     =  24u
        [ 81,   0,  81],  # ground        =  25u
        [150, 100, 100],  # bridge        =  26u
        [230, 150, 140],  # rail track    =  27u
        [180, 165, 180]   # guard rail    =  28u
])

INHOUSE_PALETTE = np.array([
        [  0,   0,   0],  # unlabeled     =   0u
        [0,  142, 0],       # obstacle          =   1u
        [128, 64, 128],  # road      =   2u

])


mask_fisheyes_carla = {"fisheye_front" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/carlaSim/roi_mask_carlaSim_fisheye_front.png",
                       "fisheye_rear" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/carlaSim/roi_mask_carlaSim_fisheye_rear.png",
                       "fisheye_left" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/carlaSim/roi_mask_carlaSim_fisheye_left.png",
                       "fisheye_right" : "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/vehicle_mask/carlaSim/roi_mask_carlaSim_fisheye_right.png"}

all_data = []

for key, value in mask_fisheyes_carla.items():
    mask = np.array(Image.open(value))
    mask_fisheyes_carla[key] = mask

for bag_name in os.listdir(DATA_ROOT):
    bag_dir = os.path.join(DATA_ROOT, bag_name)
    for topic_name in list_extracted_topics:
        image_dir = os.path.join(bag_dir, topic_name)
        albedo_dir = os.path.join(bag_dir, topic_name + ".albedo")
        depth_dir = os.path.join(bag_dir, topic_name + ".depth.logdepth")
        semantic_dir = os.path.join(bag_dir, topic_name + ".seg")
        normal_dir = os.path.join(bag_dir, topic_name + ".normal")

        position = topic_name.split('_')[-1]

        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            albedo_path = os.path.join(albedo_dir, image_name)
            depth_path = os.path.join(depth_dir, image_name.replace("jpeg", "png"))
            normal_path = os.path.join(normal_dir, image_name.replace("jpeg", "png"))
            semantic_path = os.path.join(semantic_dir, image_name.replace("jpeg", "png"))

            image = Image.open(image_path)
            albedo = Image.open(albedo_path)
            normal = Image.open(normal_path)
            depth = Image.open(depth_path)
            semantic_label = Image.open(semantic_path)
            semantic_label = np.array(semantic_label)[:, :, 0]

            image = (np.array(image)[:, :, :3] / 255.0) * 2 -1
            
            label = np.zeros((semantic_label.shape[0], semantic_label.shape[1]), dtype=np.uint8)
            label_road = np.zeros((semantic_label.shape[0], semantic_label.shape[1]), dtype=np.uint8)
            mask = mask_fisheyes_carla[topic_name]   

            for i in range(len(SYN_CARLA_INHOUSE_PALETTE)):
                if i == 1 or i == 24 or i == 25: # road
                    label[semantic_label==i] = 2 
                    label_road[semantic_label==i] = 2
                elif 12 <= i and i <= 19: # person
                    label[semantic_label==i] = 1
                else:
                    pass
            road_cnts, _ = cv2.findContours(label_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            update_label_road = cv2.fillPoly(label_road, road_cnts,  2) # 1 is the road class

            label[update_label_road!=0] = update_label_road[update_label_road!=0]
            label[mask[..., 0] == 0]=0

            masked_image = image * (mask / 255.0)
            masked_image = np.array((masked_image * 0.5 + 0.5) * 255.0, dtype=np.uint8)
            img = Image.fromarray(masked_image)
            new_image_path = os.path.join(OUT_IMG_ROOT, position + "_" + image_name)
            new_semantic_path = os.path.join(OUT_SEMANTIC_ROOT, position + "_" + image_name.replace("jpeg", "npy"))

            img.save(new_image_path)
            np.save(new_semantic_path, label)

            data = new_image_path + "," + new_semantic_path + "," + position
            data += "," + albedo_path
            data += "," + normal_path
            data += "," + depth_path
            all_data.append(data)



from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
train_data, test_data = train_test_split(all_data, test_size=0.05, random_state=42)

# Print the sizes of the splits to verify
print(f"Training data size: {len(train_data)}")
print(f"Testing data size: {len(test_data)}")

wf = open(TRAIN_TXT_PATH, "+w")
for data in train_data:
    wf.write(data + "\n")

wf.close()

wf = open(TEST_TXT_PATH, "+w")
for data in test_data:
    wf.write(data + "\n")
wf.close()