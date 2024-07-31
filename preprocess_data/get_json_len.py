import json

json_path = "/lustre/scratch/client/vinai/users/tungdt33/ARP/code/segment-arp/val_private_2024-07-01.json"


with open(json_path) as json_data:
    real_data = json.load(json_data)

print(f"LENGTH={len(real_data)}")