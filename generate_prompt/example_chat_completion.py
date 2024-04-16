# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire
from llama import Llama
from itertools import product


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    text_save_path: str,
    dataset_name: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    if dataset_name == "coco":
        # coco
        label_list = ["bus", "dog", "cow", "elephant", "umbrella", "tie", "skateboard", "cup", "knife",
                      "cake", "couch", "computer keyboard", "sink", "scissors", "airplane", "cat", "snowboard equipment"]
    else:
        # nus
        label_list = ["clouds", "bridge", "sports", "protest", "rocks", "nighttime", "surf", "leaf", "beach", 
                      "sky", "toy", "sand", "tiger", "railroad", "flowers", "fire", "snow", "garden", 
                      "sun", "food", "tower", "elk", "street", "train", "running", "fox", "military", 
                      "moon", "fish", "map", "town", "water", "sunset", "temple", "bear", "tree", 
                      "cityscape", "book", "sign", "house", "vehicle", "police", "buildings", "boats", "cars", 
                      "tattoo", "rainbow", "waterfall", "earthquake", "cow", "horses", "glacier", "plants", "animal", 
                      "whales", "plane", "swimmers", "window", "person", "cat", "wedding", "statue", "harbor", 
                      "mountain", "birds", "valley", "flags", "road", "dancing", "frost", "castle", "dog", 
                      "ocean", "grass", "computer", "zebra", "reflection", "coral", "lake", "soccer", "airport"]

    label_pair = list(product(label_list, repeat=2))
    
    while(len(label_pair) != 0):
        label = list(label_pair[0])
        label_pair.pop(0)
        if (label[1], label[0]) in label_pair:
            label_pair.remove((label[1], label[0]))
        
        full_path = text_save_path + label[0] + '_' + label[1] + '.txt'
        file = open(full_path, 'w')
        
        if label[1] != label[0]:
            condition = 'Please forget the previous keywords. Now, if you are a painter, I have a few key words for you. Please create a picture that contains both of the key words I have given. Can you describe the painting you will create in one phrase? Use "next to" between two keywords.'
            generate_prompt = 'Now, I give keywords "a {}" and "a {}", please generate 10 different phrases that satisfies the condition. And each phrase should contain no less than 10 words.'.format(label[0], label[1])
        else:
            condition = 'Please forget the previous keywords. Now, if you are a painter, I have a few key words for you. Please create a picture that contains both of the key words I have given. Can you describe the painting you will create in one phrase?'
            generate_prompt = 'Now, I give keywords "a {}", please generate 10 different phrases that satisfies the condition. And each phrase should contain no less than 10 words.'.format(label[0])
        
        dialogs = [
            [
                {"role": "user", "content": condition},
                {"role": "assistant", "content": 'Please enter keywords.'},
                {"role": "user", "content": generate_prompt},
            ]
        ]
        
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            file.write(result['generation']['content']) 
            file.close()
            print("\n===============================================\n")

if __name__ == "__main__":
    fire.Fire(main)
