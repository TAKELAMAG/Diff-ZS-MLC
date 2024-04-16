import torch
import os
# from einops import repeat
import time
import argparse
import random
import json
from torch.optim import lr_scheduler
from src.loss_function.losses import AsymmetricLoss, BCELoss
from src.model.model import Diffusion_Clip
from src.helper_functions.helper_functions import remove_duplication
from clip_classification import ClipPipeline, get_label_list
from diffusers_local.trunk.src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import logging
import random
from itertools import product
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./result",
                    nargs="?", help="dir to write results to")
    parser.add_argument("--prompt_path", type=str,
                    nargs="?", help="dir to write results to")
    parser.add_argument("-steps", type=int,default=50,
                    help="number of ddim sampling steps")
    parser.add_argument("-d", "--device", default="cuda", 
                    help="computation device to use", choices=["cpu", "cuda"])
    parser.add_argument('-mmln', '--max-mulit-label-num', default=3, type=int,
                    metavar='N', help='synthitc image label number')
    parser.add_argument('-syn_num', '--syn-num', default=20, type=int,
                    metavar='N', help='synthitc per class number')
    parser.add_argument('-lr', '--learning-rate', default=1e-6, type=float,
                    metavar='N', help='learning rate')
    parser.add_argument('-epoch', '--epoch', default=1000, type=int,
                    metavar='N', help='epoch')
    parser.add_argument('-accumulation_steps', '--accumulation-steps', default=4, type=int,
                    metavar='N', help='gradient accumulation steps')
    parser.add_argument('-train_scale', '--train-scale', default=768, type=int,
                    metavar='N', help='synthitc per class number')
    parser.add_argument('-test_scale', '--test-scale', default=768, type=int,
                    metavar='N', help='synthitc per class number')
    opt = parser.parse_args()
    
    load_dict = {
        "images": [], "annotations": [], "categories": []
    }
    all_label = get_label_list()
    categories_id = {
        "bus": 6, "dog": 18, "cow": 21, "elephant": 22, "umbrella": 28, "tie": 32, "skateboard": 41, "cup": 47, "knife": 49, 
        "cake": 61, "couch": 63, "computer keyboard": 76, "sink": 81, "scissors": 87, "airplane": 5, "cat": 17, "snowboard equipment": 36
    }
    images_num = {
        "bus": 1, "dog": 1, "cow": 1, "elephant": 1, "umbrella": 1, "tie": 1, "skateboard": 1, "cup": 1, "knife": 1, 
        "cake": 1, "couch": 1, "computer keyboard": 1, "sink": 1, "scissors": 1, "airplane": 1, "cat": 1, "snowboard equipment": 1
    }
    easy_false = [["skateboard", "snowboard equipment"], ["elephant", "cow"], ["sink", "cup"], ["scissors", "knife"]]
    
    # Load prompt
    label_list = get_label_list()
    label_pair = list(product(label_list, repeat=2))
    path = opt.prompt_path
    prompt_dict = {}
    while (len(label_pair) != 0):
        label = list(label_pair[0])
        label_pair.pop(0)
        if (label[1], label[0]) in label_pair:
            label_pair.remove((label[1], label[0]))

        file_name = label[0] + '_' + label[1]
        full_path = path + file_name + '.txt'

        if os.path.exists(full_path):
            prompt_dict[file_name] = []
            file = open(full_path, 'r')
            for line in file:
                prompt_dict[file_name].append(line.strip())
        else:
            continue

    """init diffusion and clip models"""
    clip_pipeline = ClipPipeline(device=opt.device).to(torch.float32)
    
    scheduler = EulerDiscreteScheduler.from_pretrained("./generate_image_coco/stabilityai/stable-diffusion-2", subfolder="scheduler")
    diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
        "./generate_image_coco/stabilityai/stable-diffusion-2", 
        scheduler=scheduler, 
        torch_dtype=torch.float32
    ).to(opt.device)
    
    diffusion_pipeline.unet.train()
    diffusion_pipeline.unet.enable_gradient_checkpointing()
    
    model = Diffusion_Clip(opt, clip_pipeline, diffusion_pipeline, prompt_dict)
    model.to(opt.device)
    
    """cast models to fp16"""
    clip_pipeline.to(opt.device, dtype=torch.float16)
    diffusion_pipeline.unet.to(opt.device, dtype=torch.float16)
    diffusion_pipeline.vae.to(opt.device, dtype=torch.float16)
    
    clip_pipeline.requires_grad_(False)
    diffusion_pipeline.unet.requires_grad_(False)
    
    """set lora layers"""
    unet = diffusion_pipeline.unet
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=4,
        ).cuda()

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    params = []
    text_encoder = diffusion_pipeline.text_encoder
    params.extend(list(text_encoder.parameters()))
    
    weight_decay = 1e-2
    scaler = GradScaler()
    criterion = {'ASL' : AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True), 
                 'BCE' : BCELoss(reduce=True, size_average=True)}
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate * opt.accumulation_steps, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-8)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda _: 1, last_epoch=-1)
    
    print("-------------Begin inpainting-------------")
    start = time.time()
    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.outdir + '/all', exist_ok=True)
    os.makedirs(opt.outdir + '/select', exist_ok=True)
    os.makedirs(opt.outdir + '/model', exist_ok=True)
    num = 600000
    Epoch = opt.epoch
    
    # train
    model.train()
    for epoch in range(Epoch):
        exit_ef = True
        while(exit_ef):
            classes = []
            for i in range(opt.max_mulit_label_num):
                classes.append(random.choice(all_label))
            classes = remove_duplication(classes)
            if len(classes) < opt.max_mulit_label_num:
                classes.append(random.choice(all_label))
            classes = remove_duplication(classes)
            
            exit_ef = False
            for i in range(len(easy_false)):
                if [False for item in easy_false[i] if item not in classes] == []:
                    exit_ef = True

        image_name = classes[0]
        for i in range(len(classes)):
            if i != 0:
                image_name = image_name + "_" + classes[i]

        target = [0] * 17
        for item in classes:
            index = all_label.index(item)
            target[index] = 1
        target = torch.tensor([target]).to(opt.device)

        with autocast():
            output, label, confidence, image = model(classes, image_name, num, len(classes), opt.train_scale, opt.steps)
            output = output.to(opt.device)
            loss = criterion["ASL"](output, target) / opt.accumulation_steps

        print('Loss: {:.4f}, Lr: {:.1e}'.format(loss.item(), scheduler.get_last_lr()[0]))

        model.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(params, max_norm=1)

        exit_nan = False
        for para in params:
            if torch.isnan(para.grad).any():
                exit_nan = True
                
        if exit_nan == True:
            optimizer.zero_grad()
            
        if (epoch+1) % opt.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        
        if (epoch + 1) % 100 == 0:
            dir_name = "epoch_" + str(epoch + 1)
            os.makedirs(opt.outdir + '/model/' + dir_name, exist_ok=True)
            # save model params
            save_state = {}
            for param_tensor in text_encoder.state_dict():
                save_state.update({param_tensor:text_encoder.state_dict()[param_tensor]})
            torch.save(save_state, opt.outdir + "/model/" + dir_name + "/text_encoder_param.pth")
            
        # save image
        if [False for item in label if item not in classes] == [] and len(label) == len(classes):
            try:
                file_name = str(num) + "_" + image_name
                image.save(opt.outdir + "/select/" + file_name +".jpg")
                print("Saving: " + file_name + ".jpg")
            except:
                print("Saving error!")
                continue
            new_image = {
                "license": "", 
                "file_name": file_name + ".jpg", 
                "coco_url": "", 
                "height": opt.train_scale, 
                "width": opt.train_scale, 
                "date_captured": "", 
                "flickr_url": "", 
                "id": int(num)
            }
            load_dict["images"].append(new_image)
            for i in range(len(label)):
                id = num * (10 ** i)
                new_annotation = {
                    "segmentation": [],
                    "area": 100,
                    "iscrowd": 0,
                    "image_id": int(num),
                    "bbox": [],
                    "category_id": categories_id[label[i]],
                    "id": int(id),
                    "confidence": confidence[i]
                }
        else:
            print("Not saving!")
        num += 1
    
    add_categories = [
        {"supercategory": "vehicle", "id": 6, "name": "bus"}, 
        {"supercategory": "animal", "id": 18, "name": "dog"}, 
        {"supercategory": "animal", "id": 21, "name": "cow"}, 
        {"supercategory": "animal", "id": 22, "name": "elephant"}, 
        {"supercategory": "accessory", "id": 28, "name": "umbrella"}, 
        {"supercategory": "accessory", "id": 32, "name": "tie"}, 
        {"supercategory": "sports", "id": 41, "name": "skateboard"}, 
        {"supercategory": "kitchen", "id": 47, "name": "cup"}, 
        {"supercategory": "kitchen", "id": 49, "name": "knife"}, 
        {"supercategory": "food", "id": 61, "name": "cake"}, 
        {"supercategory": "furniture", "id": 63, "name": "couch"}, 
        {"supercategory": "electronic", "id": 76, "name": "keyboard"}, 
        {"supercategory": "appliance", "id": 81, "name": "sink"}, 
        {"supercategory": "indoor", "id": 87, "name": "scissors"}, 
        {"supercategory": "vehicle", "id": 5, "name": "airplane"}, 
        {"supercategory": "animal", "id": 17, "name": "cat"}, 
        {"supercategory": "sports", "id": 36, "name": "snowboard"}
    ]
    
    for i in range(len(add_categories)):
        load_dict["categories"].append(add_categories[i])
    
    print("Write out instances_synthesis.json")
    dict_json_synthesis = json.dumps(load_dict)
    with open(opt.outdir + '/instances_synthesis.json', 'w+') as file:
        file.write(dict_json_synthesis)
    print("Done!")

    end = time.time()
    print(f"Total time: {end - start}")
