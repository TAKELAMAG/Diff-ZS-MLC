import os
import torch
import numpy as np
import copy
import torch.nn as nn
import clip_local.clip as clip
import torch.nn.functional as F


def get_label_list():
    # nus_unseen
    label_list = [
        "clouds", "bridge", "sports", "protest", "rocks", "nighttime", "surf", "leaf", "beach",
        "sky", "toy", "sand", "tiger", "railroad", "flowers", "fire", "snow", "garden",
        "sun", "food", "tower", "elk", "street", "train", "running", "fox", "military",
        "moon", "fish", "map", "town", "water", "sunset", "temple", "bear", "tree",
        "cityscape", "book", "sign", "house", "vehicle", "police", "buildings", "boats", "cars",
        "tattoo", "rainbow", "waterfall", "earthquake", "cow", "horses", "glacier", "plants", "animal",
        "whales", "plane", "swimmers", "window", "person", "cat", "wedding", "statue", "harbor",
        "mountain", "birds", "valley", "flags", "road", "dancing", "frost", "castle", "dog",
        "ocean", "grass", "computer", "zebra", "reflection", "coral", "lake", "soccer", "airport"]
    
    return label_list

def get_groundTrue(labels, true_label):
    pos_list = []
    neg = torch.ones(len(labels)).to('cuda')
    for label in true_label:
        pos = torch.zeros(len(labels))
        pos[labels.index(label)] = 1
        pos_list.append(pos.to('cuda'))
        neg = neg - pos.to('cuda')
    
    return pos_list, -neg


class ClipPipeline(nn.Module):
    def __init__(self, device):
        super(ClipPipeline, self).__init__()
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.labels = get_label_list()
        
    def forward(self, image, top_k, true_label):
        
        image_feature = (image / 2 + 0.5).clamp(0, 1) # diffusion denormalize
        MEAN = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).cuda()
        STD = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).cuda()
        image_feature = (image_feature - MEAN) / STD
        image_feature = F.interpolate(image_feature, [224, 224])

        txt = clip.tokenize(self.labels).to(self.device)
        
        image_features = self.model.encode_image(image_feature)
        text_features = self.model.encode_text(txt)
        
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * torch.cosine_similarity(image_features, text_features, dim=1)
        
        '''grouping softmax'''
        pos_list, neg = get_groundTrue(self.labels, true_label)
        softmax_all = torch.zeros(len(self.labels)).to('cuda')
        for pos in pos_list:
            target = pos + neg
            target_index = torch.where(target.pow(2) == 1)
            softmax_local = torch.gather((target.pow(2) * logits_per_image), 0, target_index[0]).softmax(-1)
            
            j = 0
            target_list = target.cpu().numpy().tolist()
            for i in range(len(target_list)):
                if target_list[i] == 1:
                    softmax_all[i] = softmax_local[j]
                elif target_list[i] == -1:
                    softmax_all[i] += softmax_local[j] / len(true_label)
                else:
                    continue
                j += 1
        
        probs = softmax_all
        probs = probs.detach().cpu().numpy().tolist()
        tmp_list = copy.deepcopy(probs)
        tmp_list.sort()
        top_k = 7
        max_num_index=[probs.index(one) for one in tmp_list[::-1][:top_k]]
        label = []
        confidence = []
        for i in range(len(max_num_index)):
            if probs[max_num_index[i]] > 0.1 and self.labels[max_num_index[i]] not in true_label:
                label.append(self.labels[max_num_index[i]])
                confidence.append("%.2f"%(probs[max_num_index[i]]))
            elif self.labels[max_num_index[i]] in true_label:
                label.append(self.labels[max_num_index[i]])
                confidence.append("%.2f"%(probs[max_num_index[i]]))
            print(f"Predicted label {self.labels[max_num_index[i]]} has the probability of {probs[max_num_index[i]] * 100} %")

        return softmax_all, label, confidence

