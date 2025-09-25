from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM , AutoTokenizer,AutoProcessor
import torch
import re
model_id = "/root/dws/MCS/Models/Phi-3.5-vision-instruct" 
device = 'cuda:3'
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map=device, 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='flash_attention_2'    
)
min_pixels = 384 * 384  # 或 256 * 28 * 28
max_pixels = 384 * 384 

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=16) 
tokenizer = AutoTokenizer.from_pretrained(model_id)
images = []
placeholder = "<|image_1|>"
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
def extract_numbers(string):
    # Find all integers and floating-point numbers in the string
    numbers = re.findall(r'\d+\.\d+|\d+', string)
    return numbers

def phi_35v_fine_reasoner(image_path: str, top_k_labels,is_saliency=False, **kwargs):

    label_index = [str(i+1) for i in range(len(top_k_labels))]
    choices_dict = {label_index[i]: top_k_labels[i] for i in range(len(top_k_labels))}

    choices = [f'{label_index[i]}: {top_k_labels[i]}\n' for i in range(len(top_k_labels))]
    choices = ', '.join(choices)
    instruction = """What is in the picture? """
    prompts = f"""Choose the most relevant scene from the following options : \n{choices}. You should directly output the answer index, for example: 1."""
    template = instruction + '\n'+prompts
    
    messages = [
        {"role": "user", "content": placeholder+f"{template}"},
    ]
    
    prompt = processor.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
    )

    inputs = processor(prompt, [image_path], return_tensors="pt").to(device) 

    generation_args = { 
        "max_new_tokens": 1000, 
        "temperature": 1, 
        "do_sample": False,
        "use_cache": False,  # 添加这一行
    } 
    with torch.no_grad():
        generate_ids = model.generate(**inputs, 
        eos_token_id=processor.tokenizer.eos_token_id, 
        past_key_values=None, 
        **generation_args
        )

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    output_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

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

def phi_35v_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL, captions, is_saliency=False, I2T=False, T2I=False):
    predict = []
    entropies_all = []  # 存储所有生成 token 的 entropy
    True_id =tokenizer.convert_tokens_to_ids('True')
    for img in image_PIL:
        messages = [
                {"role": "user", "content": placeholder+f"Does the image match the caption: <{captions}>? Please directly output True or False"},
            ]
        prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
        )

        inputs = processor(prompt, [img], return_tensors="pt").to(device) 

        with torch.no_grad():
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


def Multi_layers_fusion_phi(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        mission: str,
        retracing_ratio: float,
        vision_retracing_method:str ,
    
    ):
    self.model.layers[0].mlp.apply_memvr = True
    self.model.layers[0].mlp.starting_layer = starting_layer
    self.model.layers[0].mlp.ending_layer = ending_layer
    self.model.layers[0].mlp.entropy_threshold = entropy_threshold
    self.model.layers[0].mlp.vision_retracing_method = vision_retracing_method

    
    for layer in range(28):
        self.model.layers[layer].mlp.retracing_ratio = retracing_ratio
        self.model.layers[layer].mlp.mission = mission
Multi_layers_fusion_phi(
            self=model,
            starting_layer=5,
            ending_layer=16,
            entropy_threshold=0.75,
            mission = 'multi_text',#multi_image
            retracing_ratio=0.12,#0.05-0.35
            # retracing_ratio=0.12,#0.05-0.35
            vision_retracing_method = 'adapt',#default adapt
        )
# import os
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
# predict_index = phi_35v_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL,captions, is_saliency=True,I2T = False,T2I = True)
# end_time = time.time()
# print("Time taken:", end_time - start_time)
# print(predict_index)

# image_path = '/root/dws/MCS/Datasets/ShareGPT4V/data/gqa/images/1.jpg'
# image_PIL = Image.open(image_path).convert('RGB')
# import time
# start_time = time.time()
# top_k_labels =   ['Man skates along cement wall.', '"A skateboarder rides up a concrete wall, nearly falling off as he tries a trick."', 'A skateboarder is riding his board along the boundary stone of a parking lot.', 'The man does a trick on his skateboard on a concrete ramp.', 'A man does skateboard tricks in a parking lot at night.']
# predict_index = phi_35v_fine_reasoner(image_path,top_k_labels,I2T = True,T2I = False)
# end_time = time.time()
# print("Time taken:", end_time - start_time)
# print(predict_index)

