import torch
import os
# from einops import repeat
import time
import argparse
import random
import json
from src.model.model import Diffusion_Clip
from src.helper_functions.helper_functions import remove_duplication
from clip_classification import ClipPipeline, get_label_list
from diffusers_local.trunk.src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import logging
from itertools import product
import os

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./result",
                    nargs="?", help="dir to write results to")
    parser.add_argument("--prompt_path", type=str,
                    nargs="?", help="dir to write results to")
    parser.add_argument("--ckpt_dir", type=str,
                    nargs="?", help="dir to write results to")
    parser.add_argument("-steps", type=int,default=50,
                    help="number of ddim sampling steps")
    parser.add_argument("-d", "--device", default="cuda", 
                    help="computation device to use", choices=["cpu", "cuda"])
    parser.add_argument('-mmln', '--max-mulit-label-num', default=2, type=int,
                    metavar='N', help='synthitc image label number')
    parser.add_argument('-syn_num', '--syn-num', default=20, type=int,
                    metavar='N', help='synthitc per class number')
    parser.add_argument('-test_scale', '--test-scale', default=768, type=int,
                    metavar='N', help='synthitc per class number')
    opt = parser.parse_args()

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.outdir + '/all', exist_ok=True)
    os.makedirs(opt.outdir + '/select', exist_ok=True)
    
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
    
    # load prompt
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
    
    model = Diffusion_Clip(opt, clip_pipeline, diffusion_pipeline, prompt_dict)
    model.to(opt.device)

    print("... Loading pretrained text encoder params")
    if os.path.exists(full_path):
        param_dict = torch.load(opt.ckpt_dir)
        model.diffusion_pipeline.text_encoder.load_state_dict(param_dict)
        print("Loading successfully")
    else:
        print("The encoder parameter does not exist, use the default parameter")
    
    print("-------------Begin inpainting-------------")
    start = time.time()
    num = 600000

    model.eval()
    while(all_label):
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

        with torch.no_grad():
            with autocast():
                output, label, confidence, image = model(classes, image_name, num, len(classes), opt.test_scale, opt.steps)
        
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
                "height": opt.test_scale, 
                "width": opt.test_scale, 
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
                load_dict["annotations"].append(new_annotation)
                if images_num[label[i]] == opt.syn_num:
                    all_label.remove(label[i])
                else:
                    images_num[label[i]] += 1
        else:
            print("Not saving!")
        num += 1
        print("===================================================")
        if num % 100 == 0:
            print(images_num)
    
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
    
    print("Write out train_instances_only_synthetic.json")
    dict_json_synthesis = json.dumps(load_dict)
    with open(opt.outdir + '/train_instances_only_synthetic.json', 'w+') as file:
        file.write(dict_json_synthesis)
    print("Done!")

    end = time.time()
    print(f"Total time: {end - start}")