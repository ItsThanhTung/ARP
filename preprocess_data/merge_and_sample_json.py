import json
import random

json_list = ["/lustre/scratch/client/vinai/users/tungdt33/ARP/sampling_data/exp_6class/data.json",
             "/lustre/scratch/client/vinai/users/tungdt33/ARP/sampling_data/exp_6class_night/data.json",
             "/lustre/scratch/client/vinai/users/tungdt33/ARP/sampling_data/exp_6class_rainy/data.json",
             "/lustre/scratch/client/vinai/users/tungdt33/ARP/sampling_data/exp_6class_snowy/data.json"]

all_data = []
for json_path in json_list:
    with open(json_path) as json_data:
        all_data  += json.load(json_data)

# random_choices = random.sample(all_data, 000)

# with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/train_6k.json") as json_data:
#     random_choices  += json.load(json_data)

print(len(all_data))
with open("/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real/train_6cls_full_generated.json", '+w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)