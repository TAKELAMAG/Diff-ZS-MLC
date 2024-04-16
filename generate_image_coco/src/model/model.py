import torch.nn as nn
import random
    
    
def prompt_embedding(classes, diffusion, device, prompt_dict):
    negative_prompt = "watermark, text, poorly-drawn-eyes, poorly-drawn-fingers, bad-anatomy, ugly, extra fingers, extra limbs, missing fingers, missing limbs, deformed, mutated hands and fingers"
    index = random.randint(0, 9)
    if len(classes) == 1:
        prompt = prompt_dict[classes[0] + '_' + classes[0]][index]
    else:
        if classes[0] + '_' + classes[1] in prompt_dict.keys():
            prompt = prompt_dict[classes[0] + '_' + classes[1]][index]
        elif classes[1] + '_' + classes[0] in prompt_dict.keys():
            prompt = prompt_dict[classes[1] + '_' + classes[0]][index]
        else:
            prompt = "A photo of"
            for i in range(len(classes)):
                prompt = prompt + ' a ' + classes[i] + (' next to' if (i < (len(classes) - 1)) else '.')
    print("Prompt: " + prompt)
    prompt_embeds = diffusion._encode_prompt(prompt=prompt, 
                                         negative_prompt = negative_prompt,
                                         device=device, 
                                         do_classifier_free_guidance=True,
                                         num_images_per_prompt=1)
    return prompt_embeds


class Diffusion_Clip(nn.Module):
    def __init__(self, opt, clip_pipeline, diffusion_pipeline, prompt_dict):
        super(Diffusion_Clip, self).__init__()
        self.opt = opt
        self.diffusion_pipeline = diffusion_pipeline
        self.clip_pipeline = clip_pipeline
        self.prompt_dict = prompt_dict = prompt_dict

        
    def forward(self, classes, image_name, num, top_k, scale, steps):
        prompt_embeds = prompt_embedding(classes, self.diffusion_pipeline, self.opt.device, self.prompt_dict)
        
        image, image_feature = self.diffusion_pipeline(
            prompt_embeds = prompt_embeds, 
            height=scale, width=scale, 
            num_inference_steps=steps
        )

        try:
            file_name = str(num)
            image[0].save(self.opt.outdir +"/all/" + file_name + ".jpg")
            print("Saving: " + file_name + ".jpg")
        except:
            print("Saving error!")

        output, label, confidence = self.clip_pipeline(image_feature, top_k, classes)
        
        return output, label, confidence, image[0]
    

