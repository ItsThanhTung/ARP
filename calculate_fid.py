from cleanfid import fid

print("start calculating")
score = fid.compute_fid("/lustre/scratch/client/vinai/users/tungdt33/ARP/sampling_data/test_sampling", dataset_name="MASKED_REAL",
          mode="clean", dataset_split="custom")
print(f"FID: {score}")
# fid.make_custom_stats("MASKED_REAL", "/lustre/scratch/client/vinai/users/tungdt33/ARP/data/MASKED_REAL/test", mode="clean")
