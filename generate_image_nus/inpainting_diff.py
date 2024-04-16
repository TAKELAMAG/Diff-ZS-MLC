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
import numpy as np

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
    
    all_label = get_label_list()
    categories_id = {
        "clouds": 0, "bridge": 10, "sports": 21, "protest": 41, "rocks": 59, "nighttime": 70, "surf": 78, "leaf": 85, "beach": 92,
        "sky": 110, "toy": 112, "sand": 122, "tiger": 138, "railroad": 144, "flowers": 148, "fire": 173, "snow": 183, "garden": 188,
        "sun": 196, "food": 222, "tower": 235, "elk": 241, "street": 243, "train": 258, "running": 263, "fox": 266, "military": 270,
        "moon": 272, "fish": 278, "map": 283, "town": 306, "water": 314, "sunset": 323, "temple": 349, "bear": 353, "tree": 358,
        "cityscape": 384, "book": 387, "sign": 389, "house": 396, "vehicle": 397, "police": 402, "buildings": 417, "boats": 419, "cars": 434,
        "tattoo": 444, "rainbow": 446, "waterfall": 459, "earthquake": 484, "cow": 504, "horses": 509, "glacier": 514, "plants": 535, "animal": 538,
        "whales": 558, "plane": 563, "swimmers": 579, "window": 601, "person": 632, "cat": 648, "wedding": 656, "statue": 694, "harbor": 701,
        "mountain": 715, "birds": 721, "valley": 751, "flags": 755, "road": 769, "dancing": 787, "frost": 799, "castle": 802, "dog": 815,
        "ocean": 823, "grass": 831, "computer": 836, "zebra": 848, "reflection": 874, "coral": 906, "lake": 962, "soccer": 973, "airport": 1003}
    images_num = {
        "clouds": 1, "bridge": 1, "sports": 1, "protest": 1, "rocks": 1, "nighttime": 1, "surf": 1, "leaf": 1, "beach": 1,
        "sky": 1, "toy": 1, "sand": 1, "tiger": 1, "railroad": 1, "flowers": 1, "fire": 1, "snow": 1, "garden": 1,
        "sun": 1, "food": 1, "tower": 1, "elk": 1, "street": 1, "train": 1, "running": 1, "fox": 1, "military": 1,
        "moon": 1, "fish": 1, "map": 1, "town": 1, "water": 1, "sunset": 1, "temple": 1, "bear": 1, "tree": 1,
        "cityscape": 1, "book": 1, "sign": 1, "house": 1, "vehicle": 1, "police": 1, "buildings": 1, "boats": 1, "cars": 1,
        "tattoo": 1, "rainbow": 1, "waterfall": 1, "earthquake": 1, "cow": 1, "horses": 1, "glacier": 1, "plants": 1, "animal": 1,
        "whales": 1, "plane": 1, "swimmers": 1, "window": 1, "person": 1, "cat": 1, "wedding": 1, "statue": 1, "harbor": 1,
        "mountain": 1, "birds": 1, "valley": 1, "flags": 1, "road": 1, "dancing": 1, "frost": 1, "castle": 1, "dog": 1,
        "ocean": 1, "grass": 1, "computer": 1, "zebra": 1, "reflection": 1, "coral": 1, "lake": 1, "soccer": 1, "airport": 1}
    class_relationship = {
        'person': ['military', 'police', 'swimmers'],
        'vehicle': ['train', 'boats', 'cars', 'plane'],
        'sports': ['surf', 'running', 'dancing', 'soccer'],
        'plants': ['flowers', 'tree', 'leaf'],
        'cityscape': ['garden', 'street', 'town'],
        'water': ['ocean', 'lake', 'waterfall'],
        'buildings': ['bridge', 'airport', 'tower', 'temple', 'house', 'castle'],
        'animal': ['cat', 'fox', 'fish', 'bear', 'cow', 'birds', 'dog', 'zebra', 'tiger', 'elk', 'horses', 'whales'],
    }
    
    # Load prompt
    label_list = get_label_list()
    label_pair = list(product(label_list, repeat=2))
    path = opt.prompt_path
    prompt_dict = {}
    while (len(label_pair) != 0):
        label = list(label_pair[0])
        label_pair.pop(0)

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
    num = 100000
    Epoch = opt.epoch
    
    formatted_syn_images_filtered = []
    formatted_syn_labels_filtered = []
    # train
    model.train()
    for epoch in range(Epoch):
        exit_ef = True
        while(exit_ef):
            classes = []
            for i in range(opt.max_mulit_label_num):
                classes.append(random.choice(all_label))
            classes = remove_duplication(classes)
            
            exit_ef = False

        image_name = classes[0]
        for i in range(len(classes)):
            if i != 0:
                image_name = image_name + "_" + classes[i]

        with autocast():
            output, label, confidence, image = model(classes, image_name, num, len(classes), opt.train_scale, opt.steps)
            output = output.to(opt.device)
        
        target = [0] * 81
        for lb in classes:
            index = all_label.index(lb)
            target[index] = 1
        target = torch.tensor([target]).to(opt.device)
            
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
        if [False for item in classes if item not in label] == []:
            try:
                file_name = str(num) + "_" + image_name
                image.save(opt.outdir + "/select/" + file_name +".jpg")
                print("Saving: " + file_name + ".jpg")
            except:
                print("Saving error!")
                continue
            formatted_syn_images_filtered.append(file_name + ".jpg")
            for key in class_relationship.keys():
                for item in label:
                    if item in class_relationship[key]:
                        label.append(key)
            label = remove_duplication(label)
            print(label)
            labels = np.zeros(1006, dtype=int)
            for i in range(len(label)):
                labels[categories_id[label[i]]] = 1
                if images_num[label[i]] == opt.syn_num and label[i] in all_label:
                    all_label.remove(label[i])
                else:
                    images_num[label[i]] += 1
            formatted_syn_labels_filtered.append(labels.tolist())
        else:
            print("Not saving!")
        num += 1
        epoch += 1
        print("===================================================")
        if num % 100 == 0:
            print(images_num)
    
    end = time.time()
    print(f"Total time: {end - start}")
