import json

real_json_path = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_semantic_latent_data.json"
out_json_path = "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/DATA_ARP/real_no_none_semantic_latent_data.json"


with open(real_json_path) as json_data:
    real_data = json.load(json_data)

new_data = []

for data in real_data:
    if data["semantic_path"] == "":
        continue
    new_data.append(data)

print(f"New length {len(new_data)}")
with open(out_json_path, '+w') as f:
    json.dump(new_data, f)