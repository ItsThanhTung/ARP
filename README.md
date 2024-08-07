
## Image Domain Transfer using Generative Model 
## Setup
For this project, we used python/3.11.9, pytorch/2.2.2, cuda/11.8.0, xformers/0.0.25.post1+cu118.

### Dependencies
```bash
export DGINSTYLE_PATH=/path/to/venv/dginstyle  # change this
python3 -m venv ${DGINSTYLE_PATH}
source ${DGINSTYLE_PATH}/bin/activate

pip install -r requirements.txt
```

## :baseball: Training 
### Preparing for datasets
Using the dataset from ```/lustre/scratch/client/vinai/users/tungdt33/ARP/data/sim2realARP/real``` for training and evaluation.
First we need to run the command  
```bash
python preprocess_data/split_6k_dataset.py
```
to extract dataset into ```train_6k.json``` and ```val_6k.json```


### DreamBooth fine-tuning
We fine-tune the original latent diffusion model's U-Net on our data.
```bash
export gpu_id="0,1,2,3"
bash train.sh gpu_id
```

After the training complete, we need to convert the checkpoint into usable model. Replace ```--checkpoint_path``` and ```--output_dir``` in the file bellow.
```bash
bash convert_checkpoint_to_model.sh
```

### ControlNet training with the source domain LDM
Replace the ```PRETRAINED_PATH``` in ```train_controlnet.sh```  with ```--output_dir``` after converting to train controlnet.

Then run the following command to train ControlNet on finetuned UNet. 
```bash
export gpu_id="0,1,2,3"
bash train_controlnet.sh gpu_id
```

## :baseball: Inference
After training controlnet, we will get the trained controlnet on our data. During inference, we use this Controlnet and Original Unet (SD 2.1) to generate new images.
Replace ```PRETRAINED_PATH``` with the trained controlnet model 
We could manipulate the weather of the generated image by using ```--weather_type``` in ```sampling_controlnet.sh```.

Then run the following command to generate image and semantic label.
```bash
export gpu_id="0,1,2,3"
bash sampling_controlnet.sh gpu_id
```
