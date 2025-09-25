import sys
sys.path.append('/root/dws/MCS/Codes')
import os
import random
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from apiprompting.API.API_CLIP.main import gen_mask ,blend_mask,get_model
from qwen_vl_utils import process_vision_info
import torch
import json
import re
from typing import List
from torch import functional as F
# from Grounded_sam2 import process_imglist
from PIL import Image
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
model_id = '/root/dws/MCS/Models/Qwen2-VL-7B-Instruct'
model = Qwen2VLForConditionalGeneration.from_pretrained(model_id,torch_dtype = torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'left'
# min_pixels = 512 * 28 * 28
# max_pixels=768*28*28
min_pixels = 256 * 28 * 28
max_pixels=512*28*28
# min_pixels = 256 * 28 * 28
# max_pixels=384*28*28
processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels,use_fast=True)
#processor = AutoProcessor.from_pretrained(model_id, use_fast=True) 
# print(processor.image_processor)
processor.tokenizer = tokenizer
processor.tokenizer.padding_side = 'left'
print("Qwen2-VL model loaded")
def Multi_layers_fusion_Qwen2(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        mission: str,
        retracing_ratio: float,
        vision_retracing_method:str ,
    
    ):
    self.model.language_model.layers[0].mlp.apply_memvr = True
    self.model.language_model.layers[0].mlp.starting_layer = starting_layer
    self.model.language_model.layers[0].mlp.ending_layer = ending_layer
    self.model.language_model.layers[0].mlp.entropy_threshold = entropy_threshold
    self.model.language_model.layers[0].mlp.vision_retracing_method = vision_retracing_method
    
    
    for layer in range(28):
        self.model.language_model.layers[layer].mlp.retracing_ratio = retracing_ratio
        self.model.language_model.layers[layer].mlp.mission = mission
def extract_numbers(string):
    # Find all integers and floating-point numbers in the string
    numbers = re.findall(r'\d+\.\d+|\d+', string)
    return numbers
def get_featuremap_size(img):
    from math import sqrt
    import math
    patch_size = processor.image_processor.patch_size  # 14
    merge_size = processor.image_processor.merge_size  # 14*2 = 28
    img_pixels = img.size[0] * img.size[1]
    if img_pixels < min_pixels:
        scale = sqrt(min_pixels / img_pixels)
    elif img_pixels > max_pixels:
        scale = sqrt(max_pixels / img_pixels)
    else:
        scale = 1.0
    new_w = img.size[0] * scale
    new_h = img.size[1] * scale

    resolution_unit = patch_size * merge_size  # 28

    new_w = math.ceil(new_w / resolution_unit) * resolution_unit
    new_h = math.ceil(new_h / resolution_unit) * resolution_unit

    H_visual = new_h // patch_size
    W_visual = new_w // patch_size
    
    return H_visual, W_visual
def Qwen2_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL, captions, is_saliency=False, I2T=False, T2I=False):
    
    predict = []
    entropies_all = []  # 存储所有生成 token 的 entropy
    True_id =tokenizer.convert_tokens_to_ids('True')
    for img in image_PIL:
        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"Does the image match the caption: <{captions}>? Please directly output True or False"},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            # if is_saliency:
            #     model.model.visual.is_saliency = True
            #     H_visual, W_visual = get_featuremap_size(image_inputs[0])
            #     saliency_map = get_saliency_map(image_inputs[0], H_visual, W_visual)
            #     saliency_map = saliency_map.squeeze(0).squeeze(0).to(device) # [H, W]
            #     model.model.visual.saliency_embeds = saliency_map.unsqueeze(-1)
            
            logits = model(**inputs).logits
            
            last_token_logits = logits[:, -1, :]
            probs = torch.softmax(last_token_logits, dim=-1)

            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()  # 计算 entropy
            entropies_all.append(entropy)
            predict_scores = probs[0][True_id].item()  # 获取 True 的概率
            try:
                predict_scores = float(predict_scores)
                predict_scores = int(predict_scores * 100) / 100
            except:
                predict_scores = 0.0

            predict.append(predict_scores)

    try:
        max_score = max(predict)
        
        candidates = [i for i, score in enumerate(predict) if score == max_score]
        
        if len(candidates) == 1:
            best_index = candidates[0]
        else:
            candidate_entropies = [(i, entropies_all[i]) for i in candidates]
            best_index = min(candidate_entropies, key=lambda x: x[1])[0]

        return best_index, predict, entropies_all
    except ValueError:
        return 0, None, None
    
    
    
def Qwen2_fine_reasoner(image_path: str, top_k_labels,is_saliency=False, **kwargs):
        
    label_index = [str(i+1) for i in range(len(top_k_labels))]
    choices_dict = {label_index[i]: top_k_labels[i] for i in range(len(top_k_labels))}

    choices = [f'{label_index[i]}: {top_k_labels[i]}\n' for i in range(len(top_k_labels))]
    query_text = kwargs['query_text'] if 'query_text' in kwargs else None
    query_text = query_text.replace('<|image_1|>', '') if query_text else None
    try:
        with torch.no_grad():
            choices = ', '.join(choices)
            instruction = (            
                """What is in the picture? """
                # "What type of news is this image related to?"
            )
            prompts = f"""Choose the most relevant scene from the following options : \n{choices}. You should directly output the answer index, for example: 1."""
            if query_text is not None:
                instruction = query_text
                template = instruction + prompts
            else:
                template = instruction + "\n" + prompts
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": template.replace('\n', '').replace('Represent the given image with the following question: ', '')},
                        ],
                    }
                ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            # if is_saliency:
            #     model.model.visual.is_saliency = True
            #     H_visual, W_visual = get_featuremap_size(image_inputs[0])
            #     saliency_map = get_saliency_map(image_inputs[0], H_visual, W_visual)
            #     saliency_map = saliency_map.squeeze(0).squeeze(0).to(device) # [H, W]
            #     model.model.visual.saliency_embeds = saliency_map.unsqueeze(-1)
            inputs = inputs.to(device)                
            generated_ids = model.generate(**inputs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            predict_captions = []
            # print(output_text)
            for output in output_text:
                output = extract_numbers(output.strip().split('\n')[-1])
                if len(output) > 0:
                    if output[0] in list(choices_dict.keys()):
                        predict_captions.append(choices_dict[output[0]])
            if len(predict_captions) > 0:
                predict_caption = max(set(predict_captions), key=predict_captions.count)
            else:
                predict_caption = None
            return predict_caption
    except Exception as e:
        print(f"Error during inference: {e}")
        return None
    
    
Multi_layers_fusion_Qwen2(
            self=model,
            starting_layer=5,
            ending_layer=16,
            entropy_threshold=0.75,
            mission = 'multi_text',#multi_image
            retracing_ratio=0.12,#0.05-0.35
            # retracing_ratio=0.12,#0.05-0.35
            vision_retracing_method = 'adapt',#default adapt
        )    
    
# path = '/root/dws/MCS/Datasets/ShareGPT4V/data/gqa/images'
# image_path = [os.path.join(path,f'{i}.jpg') for i in range(1,6)]
# image_PIL = []
# for image in image_path:
#     img = Image.open(image)
#     image_PIL.append(img)
# i =0 
# captions = " A man with glasses is wearing a beer can crocheted hat."
# import time
# start_time = time.time()
# predict_index = Qwen2_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL,captions, is_saliency=True,I2T = False,T2I = True)
# end_time = time.time()
# print("Time taken:", end_time - start_time)
# print(predict_index)


# image_path = '/root/dws/MCS/Datasets/ShareGPT4V/data/gqa/images/1.jpg'
# image_PIL = Image.open(image_path).convert('RGB')
# import time
# start_time = time.time()
# captions =   ['Man skates along cement wall.', '"A skateboarder rides up a concrete wall, nearly falling off as he tries a trick."', 'A skateboarder is riding his board along the boundary stone of a parking lot.', 'The man does a trick on his skateboard on a concrete ramp.', 'A man does skateboard tricks in a parking lot at night.']
# predict_index = Qwen2_fine_reasoner(image_PIL,captions,I2T = True,T2I = False)
# end_time = time.time()
# print("Time taken:", end_time - start_time)
# print(predict_index)