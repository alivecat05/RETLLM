from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import torch.nn.functional as F
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import random
import numpy as np
from accelerate import Accelerator

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 避免 cudnn 引入随机性（仅对某些操作）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)  
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
model_id = '/root/dws/MCS/Models/Qwen2.5-VL-7B-Instruct'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,torch_dtype = torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'left'
min_pixels = 256 * 28 * 28
max_pixels=512*28*28
processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels,use_fast=True)
processor.tokenizer = tokenizer
processor.tokenizer.padding_side = 'left'



def Multi_layers_fusion(
        self,
        starting_layer: int,
        ending_layer: int,
        entropy_threshold: float,
        mission: str,
        retracing_ratio: float,
        vision_retracing_method:str ,
        is_fusion: bool,
        selected_layer: list ,
    
    ):
    self.model.language_model.layers[0].mlp.apply_memvr = True
    self.model.language_model.layers[0].mlp.starting_layer = starting_layer
    self.model.language_model.layers[0].mlp.ending_layer = ending_layer
    self.model.language_model.layers[0].mlp.entropy_threshold = entropy_threshold
    self.model.language_model.layers[0].mlp.vision_retracing_method = vision_retracing_method
    
    self.model.visual.is_fusion = is_fusion
    self.model.visual.selected_layer = selected_layer
    
    for layer in range(28):
        self.model.language_model.layers[layer].mlp.retracing_ratio = retracing_ratio
        self.model.language_model.layers[layer].mlp.mission = mission
        
Multi_layers_fusion(
            self=model,
            starting_layer=5,
            ending_layer=16,
            entropy_threshold=0.75,
            mission = 'multi_text',#multi_image
            # mission = 'multi_image',#multi_image
            retracing_ratio=0.12,#0.05-0.35
            # retracing_ratio=0.12,#0.05-0.35
            vision_retracing_method = 'adapt',#default adapt
            is_fusion=False,
            selected_layer=[4,21,23,23] 
        )


path = '/root/dws/MCS/Datasets/625_coco_images/Group15792'
# # image_path = [os.path.join(path,f'topk_image_index_{i}.jpg') for i in range(5)]
image_path = [os.path.join(path,f'{i}.jpg') for i in range(5)]
image_PIL = []
for image in image_path:
    img = Image.open(image)
    image_PIL.append(img)
i =0 
# captions = "A toddler sleeping on a couch with a video game controller in his hand."

def Qwen_forward(captions, image_PIL,shuffled=False):
    instruction = (

                f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""
            )
    if shuffled:
        messages = [
                        {
                            "role": "user",
                            "content": [
                                {'type': 'text', 'text': f'{image_PIL[0]["index"] + 1}'},
                                {"type": "image", "image":  image_PIL[0]["image"]},
                                
                                {'type': 'text', 'text': f'{image_PIL[1]["index"] + 1}'},
                                {"type": "image", "image":  image_PIL[1]["image"]},
                                
                                {'type': 'text', 'text': f'{image_PIL[2]["index"] + 1}'},
                                {"type": "image", "image":  image_PIL[2]["image"]},
                                
                                {'type': 'text', 'text': f'{image_PIL[3]["index"] + 1}'},
                                {"type": "image", "image":  image_PIL[3]["image"]},
                                
                                {'type': 'text', 'text': f'{image_PIL[4]["index"] + 1}'},
                                {"type": "image", "image":  image_PIL[4]["image"]},
                                {"type": "text", "text": instruction}
                            ],
                        }
                    ]
    else:
        messages = [
                {
                    "role": "user",
                    "content": [
                        {'type': 'text', 'text': '1'},
                        {"type": "image", "image": image_PIL[0]},
                        {'type': 'text', 'text': '2'},
                        {"type": "image", "image": image_PIL[1]},
                        {'type': 'text', 'text': '3'},
                        {"type": "image", "image": image_PIL[2]},
                        {'type': 'text', 'text': '4'},
                        {"type": "image", "image": image_PIL[3]},
                        {'type': 'text', 'text': '5'},
                        {"type": "image", "image": image_PIL[4]},
                        {"type": "text", "text": instruction}
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
    inputs = inputs.to(device)
    
    logits = model(**inputs).logits
    
    last_token_logits = logits[:, -1, :]
    probs = F.softmax(last_token_logits, dim=-1)

    top_k = 5
    top_probs, top_indices = torch.topk(probs, top_k)
    token = []
    for i in range(top_k):
        token_id = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        print(f"Token: {tokenizer.decode(token_id)}, Probability: {prob:.4f}")
        token.append(tokenizer.decode(token_id))
    print('-' * 30)
    del inputs, logits, last_token_logits
    torch.cuda.empty_cache()
    def find_first_integer(lst):
        for item in lst:
            if item.isdigit():
                return int(item)
        return 1
    
    pred = find_first_integer(token)
    return probs

    
def zero_temperature(logit_tensor,percentile=0.2):
    log_probs = torch.log(logit_tensor + 1e-10)
    per_view_entropy = -torch.sum(logit_tensor * log_probs, dim=-1) 

    entropy_sorted_indices = torch.argsort(per_view_entropy)
    num_views = logit_tensor.shape[0]
    k = int(num_views * percentile)
    selected_indices = entropy_sorted_indices[:k]  
    selected_probs = logit_tensor[selected_indices]  
    # selected_probs = logit_tensor 
    zero_ed = []
    temperature= torch.finfo(selected_probs.dtype).eps
    for l in selected_probs:
        prob = F.softmax(l/temperature, dim=0)
        zero_ed.append(prob)
        # print(prob)
    zero_ed = torch.stack(zero_ed, dim=0)
    zero_ed  =zero_ed.sum(dim=0)

    # pred = torch.argmax(zero_ed).item()
    # pred = tokenizer.decode(pred)
    top_k = 5
    top_probs, top_indices = torch.topk(zero_ed, top_k)
    token = []
    for i in range(top_k):
        token_id = top_indices[i].item()
        prob = top_probs[i].item()
        print(f"Token: {tokenizer.decode(token_id)}, Probability: {prob:.4f}")
        token.append(tokenizer.decode(token_id))
    print('-' * 30)
    def find_first_integer(lst):
        for item in lst:
            if item.isdigit():
                return int(item)
        return 1
    pred = find_first_integer(token)
    return pred
   


def ZERO_tta(captions, image_PIL, shuffled_idxes,k=32):
    
    all_probs = []    
    with torch.no_grad():
        probs_original = Qwen_forward(captions, image_PIL, shuffled=False)
        all_probs.append(probs_original)
        shuffled_images_with_index = [] 
        # [{'index': idx, 'image': image_PIL[idx]} for shuffled_idx in shuffled_idxes for idx in shuffled_idx]
        for shuffled_idx in shuffled_idxes:
            s_idx = []
            for idx in shuffled_idx:
                s_idx.append({'index': idx, 'image': image_PIL[idx]})
            shuffled_images_with_index.append(s_idx)
        
        for i in range(k - 1):
            probs = Qwen_forward(captions, shuffled_images_with_index[i], shuffled=True)
            all_probs.append(probs)
            
        all_probs = torch.cat(all_probs, dim=0)
        pred = zero_temperature(all_probs)

        return pred



k=32
shuffled_idxes =  [torch.randperm(len(image_PIL)).tolist() for _ in range(k)] 
print(f'Shuffled Indices: {shuffled_idxes}')

    
# if __name__ == "__main__":

#     pred=ZERO_tta(captions, image_PIL, shuffled_idxes,k=k)
    
#     print(f'Predicted Index: {pred}')