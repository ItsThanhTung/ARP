import torch
from diffusers import StableDiffusionPipeline    
from PIL import Image             
from diffusers import DPMSolverMultistepScheduler                                                                                                                                                                                              
                                                                                                                                                                                                                                              
                                                                                                                                                                                   


pipe = StableDiffusionPipeline.from_pretrained("/lustre/scratch/client/vinai/users/tungdt33/ARP/code/ARP/exp_real_data_2.1/model-45000") 
pipe.scheduler = DPMSolverMultistepScheduler .from_config(pipe.scheduler.config)         

prompt = ""
pipe = pipe.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)   
image = pipe(prompt, generator=generator).images[0]       

pipeline_args = {"prompt" : prompt, "width" : 640, "height" : 480}

with torch.inference_mode():
    image = pipe(**pipeline_args, num_inference_steps=25, generator=generator).images[0]

image.save("/lustre/scratch/client/vinai/users/tungdt33/ARP/test.png")