import os 
import json 
import random 

real_json_path = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_no_none_semantic_latent_data.json"
synthetic_json_path = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/synthetic_train_latent_data.json"
out_json_path = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_20K_synthetic_no_none_semantic_latent_data.json"


with open(real_json_path) as json_data:
    new_data = json.load(json_data)

with open(synthetic_json_path) as json_data:
    synthetic_data = json.load(json_data)

random.shuffle(synthetic_data) 
for i, data in enumerate(synthetic_data):
    if i > 20000:
        break
    new_data.append(data)

with open(out_json_path, '+w') as f:
    json.dump(new_data, f)
