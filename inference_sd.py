import torch
from diffusers import StableDiffusionPipeline    
from PIL import Image             
from diffusers import DPMSolverMultistepScheduler       
import os
                                                                                                                                                                                                                                             
                                                                                                                                                                      
POSITION = ["front", "rear", "left", "right"]
out_dir = "/lustre/scratch/client/vinai/users/tungdt33/ARP/sampling_data/test_images_no_cfg"
os.makedirs(out_dir, exist_ok=True)


pipe = StableDiffusionPipeline.from_pretrained("exp_real_data_1.5_prompt_no_cfg/model-60000") 
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)         


pipe = pipe.to("cuda")

with torch.inference_mode():
    for idx in range(5000):
        for pos in POSITION:
            prompt = f"A photo taken by a fisheye camera mounted on the {pos} of a car"
            image = pipe(prompt, width=640, height=400, guidance_scale=2.0).images[0]
            image.save(os.path.join(out_dir, "{}_{:06}.png".format(pos, idx)))