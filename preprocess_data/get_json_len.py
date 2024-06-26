import json

json_path = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_semantic_latent_data.json"


with open(json_path) as json_data:
    real_data = json.load(json_data)

print(f"LENGTH={len(real_data)}")