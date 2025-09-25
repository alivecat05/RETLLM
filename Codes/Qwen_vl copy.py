import sys
sys.path.append('/root/dws/MCS/Codes')
import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
# from dino_heal import get_saliency_map
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from apiprompting.API.API_CLIP.main import gen_mask ,blend_mask,get_model
from qwen_vl_utils import process_vision_info
import torch
import json
import re
from typing import List
from torch import functional as F
# from Grounded_sam2 import process_imglist
from PIL import Image
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
model_id = '/root/dws/MCS/Models/Qwen2.5-VL-7B-Instruct'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,torch_dtype = torch.bfloat16).to(device).eval()
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
print("Qwen2.5-VL model loaded")



def Multi_layers_fusion(
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
        

# clip_model, prs, clip_preprocess, device, clip_tokenizer=get_model(model_name = 'ViT-L-14-336', layer_index=22,device = device)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 避免 cudnn 引入随机性（仅对某些操作）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)  
def Qwen_fine_reasoner_Retrieval_scores_batch(image_PIL: list, captions: list, I2T=False, T2I=False):
    try:
        prompt_template = """
        You are a professional model for evaluating how well a caption describes an image.
        Given:
        - A caption: {caption}
        - An image
        You must assess the match between the caption and image using **7 precise criteria**. For each criterion:
        - Provide a score between 0 and 1 (e.g., 0.7352)
        - Include a **brief explanation** in one sentence
        - All scores must have **exactly 4 decimal places**
        ---
        ### Scoring Criteria (JSON keys and explanation format):
        - object_presence: Is each mentioned object present and clearly visible?
        - count_consistency: Do object quantities match?
        - attribute_consistency: Do properties like color, size, or shape match?
        - action_consistency: Are described actions visible in image?
        - spatial_relationship: Are spatial positions (e.g., “on”, “under”) accurately represented?
        - interaction: Are interactions between objects or people depicted?
        - scene_context: Does the overall scene or setting match the caption?
        - relevance_score: How relevant is the caption to the image?
        ---
        Return a JSON object strictly in the following format:
        {{
            "object_presence": {{"score": <float>, "explanation": "<short sentence>"}},
            "count_consistency": {{"score": <float>, "explanation": "<short sentence>"}},
            "attribute_consistency": {{"score": <float>, "explanation": "<short sentence>"}},
            "action_consistency": {{"score": <float>, "explanation": "<short sentence>"}},
            "spatial_relationship": {{"score": <float>, "explanation": "<short sentence>"}},
            "interaction": {{"score": <float>, "explanation": "<short sentence>"}},
            "scene_context": {{"score": <float>, "explanation": "<short sentence>"}},
            "relevance_score": {{"score": <float>, "explanation": "<short sentence>"}},
        }}

        """
        prompt = prompt_template.format(caption =captions[0]) 
        prompts = [prompt for _ in range(len(image_PIL))]
        messages = [
            [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]}
            ]
            for img, prompt in zip(image_PIL, prompts)
        ]

        texts, image_inputs= [], []
        for msg in messages:
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            img_input, _ = process_vision_info(msg)
            image_inputs.append(img_input[0])


        inputs = processor(text=texts,images=image_inputs,padding=True,return_tensors="pt")

        with torch.no_grad(),torch.inference_mode(): 
            outputs = model.generate(**inputs, max_new_tokens=512, top_k=50, top_p=0.95, repetition_penalty=1)
        
            
        generated_ids_trimmed = [ out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], outputs)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        scores = []
        for text in output_texts:
            result_json = extract_json_from_string(text)
            score = consistency_dict(result_json,tau=0.9)
            scores.append(score)

        scores_soft = torch.softmax((torch.tensor(scores)) / 0.1, dim=0).tolist()
        max_index = scores_soft.index(max(scores_soft))
        return max_index
    except Exception as e:
        print(f"Error: {e}")
        return 0 

def compute_entropy(probs):
    """计算熵（使用自然对数）"""
    # 避免 log(0)，对零概率贡献为 0
    probs = np.array(probs)
    non_zero_probs = probs[probs > 0]
    if len(non_zero_probs) == 0:
        return 0.0
    entropy = -np.sum(non_zero_probs * np.log(non_zero_probs + 1e-10))  # 加小值避免数值问题
    return entropy
   
def calculate_entropy(logits):
    entropys = []
    for l in logits:
        probs = torch.softmax(l, dim=-1)
        log_probs = torch.log(probs + 1e-10)  # 防止 log(0)
        entropy = -torch.sum(probs * log_probs, dim=-1).item()
        entropys.append(entropy)
    return entropys
def get_explanation_scores(parsed_dict):
    score_values = {}

    for key, value in parsed_dict.items():
        if isinstance(value, dict) and "score" in value:
            # 新格式：{"score": float, "explanation": str}
            score = value["score"]
        elif isinstance(value, (int, float)):
            # 旧格式：直接是 float 值
            score = value
        else:
            score = 0.0  # fallback
        score_values[key] = float(score)

    return score_values

def match_answer(string):
    if isinstance(string, list):
        scores = []
        for item in string:
            if isinstance(item, str):
                string = item
                match = re.search(r'Answer:\s*([0-9]*\.?[0-9]+)', string)
                if match:
                    score = float(match.group(1))
                    scores.append(score)
        return scores
    match = re.search(r'Answer:\s*([0-9]*\.?[0-9]+)', string)

    if match:
        score = float(match.group(1))
        # print(score)  # 输出：0.8732
        return score
    else:
        print("没有找到分数")
        return 0

def extract_numbers(string):
    # Find all integers and floating-point numbers in the string
    numbers = re.findall(r'\d+\.\d+|\d+', string)
    return numbers

def consistency_dict(predict_dict:dict,tau):
    
    compute_score =0
    if predict_dict is not None:
        c_scores = get_explanation_scores(predict_dict)
        for key in c_scores.keys():
            if key !='relevance_score':
                compute_score += float(c_scores[key])*0.1
        relevance_score = c_scores['relevance_score']

        computed_score = compute_score +relevance_score
        return computed_score
def consistency_penalty(predict_dict: dict, tau):
    penalty = []
    if predict_dict is not None:
        c_scores = get_explanation_scores(predict_dict)
        for key in c_scores.keys():
            if key != 'relevance_score':
                pen = 1-float(c_scores[key]) 
                penalty.append(pen)
        relevance_score = c_scores['relevance_score']
        avg_penalty = sum(penalty) / len(penalty)       
        final_scores = relevance_score * (1 - tau * avg_penalty)
        # accelerator.print(f"Final scores: {final_scores}")
        return final_scores
def extract_json_from_string(text: str) -> dict:

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("未找到有效的 JSON 对象。")

    json_str = match.group()

    # 清理 JSON 中最后一个元素后的多余逗号
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("JSON 解码失败，原始内容如下：\n", json_str)
        raise e

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

def extract_attention_weights(model_outputs):
    """从模型输出中提取注意力权重"""
    if hasattr(model_outputs, 'attentions') and model_outputs.attentions is not None:
        # 返回最后一层的注意力权重
        last_layer_attention = model_outputs.attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
        return last_layer_attention
    return None

def get_vision_attention_map(attention_weights, vision_token_start, vision_token_end, H_visual, W_visual):
    """
    从注意力权重中提取视觉tokens的注意力图
    attention_weights: [batch_size, num_heads, seq_len, seq_len]
    vision_token_start: 视觉token在序列中的起始位置
    vision_token_end: 视觉token在序列中的结束位置
    """
    if attention_weights is None:
        return None
    
    # 取最后一个token对视觉tokens的注意力
    last_token_attention = attention_weights[0, :, -1, vision_token_start:vision_token_end]  # [num_heads, num_vision_tokens]
    
    # 对所有头取平均
    avg_attention = torch.mean(last_token_attention, dim=0)  # [num_vision_tokens]
    
    # 重塑为图像形状
    attention_map = avg_attention.view(H_visual, W_visual)
    
    return attention_map

def visualize_attention_on_image(image, attention_map, save_path=None, alpha=0.6):
    """
    将注意力图叠加到原图像上进行可视化
    image: PIL图像
    attention_map: 注意力权重图 [H, W]
    save_path: 保存路径
    alpha: 注意力图的透明度
    """
    if attention_map is None:
        print("注意力图为空，无法可视化")
        return None
    
    # 将注意力图转换为numpy数组并归一化
    attention_np = attention_map.cpu().numpy()
    attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)
    
    # 将图像转换为numpy数组
    img_np = np.array(image)
    
    # 将注意力图resize到原图像大小
    attention_resized = cv2.resize(attention_np, (img_np.shape[1], img_np.shape[0]))
    
    # 创建热力图
    plt.figure(figsize=(12, 6))
    
    # 显示原图
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # 显示注意力叠加图
    plt.subplot(1, 2, 2)
    plt.imshow(img_np)
    
    # 创建热力图colormap
    cmap = plt.cm.hot
    attention_colored = cmap(attention_resized)
    
    plt.imshow(attention_colored, alpha=alpha)
    plt.title('Attention Visualization')
    plt.axis('off')
    
    # 添加colorbar
    im = plt.imshow(attention_resized, cmap='hot', alpha=0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"注意力可视化保存到: {save_path}")
    
    plt.show()
    
    return attention_resized

def save_attention_heatmap(attention_map, save_path):
    """单独保存注意力热力图"""
    if attention_map is None:
        return
    
    attention_np = attention_map.cpu().numpy()
    attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min() + 1e-8)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_np, cmap='hot', interpolation='bilinear')
    plt.colorbar()
    plt.title('Attention Heatmap')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"注意力热力图保存到: {save_path}")
    
    plt.show()



def Qwen_fine_reasoner_Retrieval_scores_batch_2(image_PIL: list, captions: list, I2T=False, T2I=False):
    try:
        prompt_template = """
                Does the image match the caption: <{caption}>? Please directly output the relevance score in a format of "Answer: score" Keep four decimal places

                """
        prompts = [prompt_template.format(caption=captions) for _ in range(len(image_PIL))]
        messages = [
            [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]}
            ]
            for img, prompt in zip(image_PIL, prompts)
        ]

        texts, image_inputs = [], []
        for msg in messages:
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            img_input, _ = process_vision_info(msg)
            image_inputs.append(img_input[0])

        inputs = processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(device)
        with torch.no_grad(),torch.inference_mode(): 
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], outputs)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        scores = []
        for text in output_texts:
            result_json = extract_json_from_string(text)
            score = consistency_penalty(result_json, tau=0.05)
            scores.append(score)
        scores = torch.softmax((torch.tensor(scores)) / 0.1, dim=0).tolist()
        max_index = scores.index(max(scores))
        return max_index
    except Exception as e:
        print(f"Error: {e}")
        return 0

def Qwen_fine_reasoner_Retrieval_scores(image_PIL:list,captions:list, I2T =False,T2I = False):
    predict = []
    # IMG = Image.open(d_path)
    for img in image_PIL:
        with torch.no_grad():
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            # {"type": "image", "image": IMG},
                            {"type": "text", "text": f"""Does the image match the caption: <{captions}>? Please directly output the relevance score in a format of "Answer: score" Keep four decimal places"""},
                            # {"type": "text", "text": f"""Does the image match the caption: <{captions}>? Please output the relevant score directly. For example: 0.7256."""},
                            # {"type": "text", "text": f"""Does the image match the caption: <{captions}>? Please output the relevant score directly within the score tag. For example: <score>0.7256</score>."""},
                            # {"type": "text", "text": f"""Dose the image strongly match the caption:{captions}? Please perform the task by first stating your intent as a high-level plan for approaching the problem or task. Then, provide the step-by-step reasoning or generation process, ensuring it aligns with the stated intent .You should also analyze the weaknesses  Please output the relevance score in a format of "Answer: score" Keep twe decimal places"""},
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
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=77,
                output_logits=True,
                return_dict_in_generate=True
            )
            
            logits = generated_ids.logits
                        
            probs = [torch.softmax(log, dim=-1) for log in logits]


            generated_ids = generated_ids.sequences
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # probs = extract_numbers(output_text[0])
            probs = match_answer(output_text[0])
            # print(output_text[0])
            # probs=re.search(r'\d+', output_text[0]).group()
            print(probs)
            try:
                # predict_scores  = float(probs[0])
                predict_scores  = float(probs)
                predict_scores=int(predict_scores * 100) / 100
            except:
                predict_scores = 0.0
            # probs = match_answer(output_text[0])
            # predict_scores  = probs
            predict.append(predict_scores)
            
    try:
        max_index = predict.index(max(predict))
        return max_index,predict
    except:
        return 0,None
     
def Qwen_fine_reasoner_Retrieval_scores_batch_entropy(image_PIL: list, captions: list, I2T=False, T2I=False):
    predict = []
    entropies_all = []  # 存储每个样本的平均 entropy

    # 构建 messages 列表
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"""Does the image match the caption: <{captions}>? Please directly output the relevance score in a format of "Answer: score" Keep four decimal places"""},
                ],
            }
        ]
        for img in image_PIL
    ]

    texts, image_inputs = [], []
    for msg in messages:
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        img_input, _ = process_vision_info(msg)
        image_inputs.append(img_input[0])

    with torch.no_grad():
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=77,
            output_logits=True,
            return_dict_in_generate=True
        )

        logits_list = generated_ids.logits  # tuple of tensors
        sequences = generated_ids.sequences


        batch_size = inputs.input_ids.size(0)
        entropy_sum = [0.0] * batch_size
        token_count = [0] * batch_size

        # 遍历每个 step 的 logits
        for logits_step in logits_list:
            # logits_step shape: [batch_size, vocab_size]
            entropies = calculate_entropy(logits_step)  # list of float, length == batch_size

            for i, entropy in enumerate(entropies):
                entropy_sum[i] += entropy
                token_count[i] += 1

        # 计算每个样本的平均 entropy
        avg_entropies = [s / c if c > 0 else 0.0 for s, c in zip(entropy_sum, token_count)]
        entropies_all.extend(avg_entropies)
        entropies_all = [round(entropy, 4) for entropy in entropies_all]  # 保留四位小数
        # 解码生成结果
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, sequences)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        probs = match_answer(output_text)
        try:
            predict_scores = [float(prob) for prob in probs]
            predict_scores = [int(p * 100) / 100 for p in predict_scores]
        except Exception as e:
            print(f"Error parsing scores: {e}")
        predict = predict_scores

    try:
        max_score = max(predict_scores)
        
        candidates = [i for i, score in enumerate(predict_scores) if score == max_score]
        
        if len(candidates) == 1:
            best_index = candidates[0]
        else:
            candidate_entropies = [(i, entropies_all[i]) for i in candidates]
            best_index = min(candidate_entropies, key=lambda x: x[1])[0]

        return best_index, predict_scores, entropies_all
    except ValueError:
        return 0, None, None
    
def Qwen_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL, captions, is_saliency=False, I2T=False, T2I=False, 
                                                    ):

    
    predict = []
    entropies_all = []  # 存储所有生成 token 的 entropy
    True_id = tokenizer.convert_tokens_to_ids('True')
    
    for idx, img in enumerate(image_PIL):
        with torch.no_grad():
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        # {"type": "text", "text": "Query image"},
                        # {"type": "image", "image": Image.open(captions[1]).convert("RGB")},
                        # {"type": "text", "text": "Candidate image"},
                        # {"type": "image", "image": Image.open(img[0]).convert("RGB")},
                        # {"type": "image", "image": img},
                        # {"type": "text", "text": f"Does the image match the caption: <{captions}>? Please directly output True or False"},
                        #{"type": "text", "text": f"Does the candidate image meet the needs of the instruction: <{captions[0]}> and query image? Please directly output True or False"},
                        #{"type": "text", "text": f"Does the image and image caption :{img[1]} are match the Query image and the Query caption :{captions[0]}? Please directly output True or False"},
                    ],
                }
            ]
            if captions[1] is not None and isinstance(captions[1], Image.Image):
                messages[0]["content"].append({"type": "text", "text": "Query image"},)
                messages[0]["content"].append({"type": "image", "image": captions[1]})
            if isinstance(img[0], Image.Image):
                messages[0]["content"].append({"type": "text", "text": "Candidate image"})
                messages[0]["content"].append({"type": "image", "image": img[0]})
            if captions[0] is not None and img[1] is not None:
                messages[0]["content"].append({"type": "text", "text": f"Does the Candidate image and Candidate image caption :{img[1]} are match the Query image and the Query caption :{captions[0]}? Please directly output True or False"})
            if captions[0] is not None and img[1] is None:
                messages[0]["content"].append({"type": "text", "text": f"Does the candidate image match the Query text: <{captions[0]}> and query image? Please directly output True or False"})

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
            
            # 获取模型输出，包括注意力权重
            logits = model(**inputs).logits
            attention_weights = None
            
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

def Qwen_fine_reasoner_Retrieval_scores_entropy(image_PIL: list, captions: list, is_saliency=False, I2T=False, T2I=False):
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

            
            
            
            logits = model(**inputs).logits
            
            last_token_logits = logits[:, -1, :]
            probs = torch.softmax(last_token_logits, dim=-1)

            top_k = 2
            top_probs, top_indices = torch.topk(probs, top_k)
            token = []
            for i in range(top_k):
                token_id = top_indices[0, i].item()
                prob = top_probs[0, i].item()
                print(f"Token: {tokenizer.decode(token_id)}, Probability: {prob:.4f}")
                token.append(tokenizer.decode(token_id))

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

def Qwen_forward(captions, image_PIL):
    instruction = (

                f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""
            )
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
    probs = torch.softmax(last_token_logits, dim=-1)

    top_k = 5
    top_probs, top_indices = torch.topk(probs, top_k)
    for i in range(top_k):
        token_id = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        print(f"Token: {tokenizer.decode(token_id)}, Probability: {prob:.4f}")
    print('-' * 30)
    pred = tokenizer.decode(torch.argmax(probs, dim=-1))
    return int(pred)-1
   
    
def Qwen_fine_reasoner_Retrieval_scores_logits(image_PIL:list,captions:list, I2T =False,T2I = False):
    
    predict = []

    for img in image_PIL:
        with torch.no_grad():
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": f"""Does the image match the caption: <{captions}>? Please directly output True or False?"""},
                            
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
            start_index = torch.where(inputs['input_ids'][0] == 25)[0][-1] + 1   ### 25 is the index of :

            temp = inputs['input_ids'][0][-5:]       ###Sys prompt   user
            temp1 = inputs['input_ids'][0][start_index:-5]      ### caption

            inputs['input_ids'][0][start_index:] = torch.cat([temp, temp1], dim=-1) 
            inputs = inputs.to(device)
            generated_ids = model(**inputs)
            logits = generated_ids['logits'][0][(start_index+8):-1]
            logits = logits.softmax(dim=-1) 
            
    try:
        # print(predict)
        max_index = predict.index(max(predict))
        return max_index,predict
    except:
        return 0
def generate_clause(caption):
    
    with torch.no_grad():
            messages = [
                    {
                        "role": "user",
                        "content": [ {"type": "text", "text": f"""{caption} Break this sentence into three clauses"""},],
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
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            sentences = re.findall(r'[^.!?;,]+[.!?;,]', output_text[0])
            sentences = [sentence.strip() for sentence in sentences]
            
            return sentences
def Qwen_fine_reasoner_Retrieval_scores_slice(image_PIL:list,captions:list, I2T =False,T2I = False):
    sentences = generate_clause(captions)
    sub1,sub2,sub3 = None,None,None
    sub1 = sentences[0] if sentences[0] else None
    sub2 = sentences[1] if len(sentences)>0 else None
    if len(sentences)>2:
        sub3 = sentences[2]
    else:
        sub3 = None    
    instruction = f"""Given a caption:{captions} and three clauses: <{sub1}>,<{sub2}>,<{sub3}>.You need to score the relevance of the caption to the image and the relevance of each clause to the image.
                    Is there any content of the clause: <{sub1}> in the picture?  score:<0-1>
                    Is there any content of the clause: <{sub2}> in the picture?  score:<0-1>
                    Is there any content of the clause: <{sub3}> in the picture?  score:<0-1>
                    Does the image match the caption: <{captions}>?               score:<0-1>
                    Finally, output the sum of the four scores in a format of "Answer: score" Keep three decimal places"""
    predict = []
    sub1,sub2,sub3=None,None,None
    for img in image_PIL:
        with torch.no_grad():
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": f"""{instruction}"""},

                            
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
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            predict_scores = match_answer(output_text[0])
            predict.append(predict_scores)
            
    try:

        max_index = predict.index(max(predict))
        return max_index,predict
    except:
        return 0
    
def Qwen_fine_reasoner_Retrieval_scores_check(image_PIL:list,captions:list, I2T =False,T2I = False):
    
    # label_index = [str(i+1) for i in range(len(image_PIL))]
    # choices = [f'{label_index[i]}: {image_PIL[i]}' for i in range(len(image_PIL))]
    # choices = ''.join(choices)
    predict = []
    try:
        for img in image_PIL:
            with torch.no_grad():
                messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": f"""Does the image match the caption: <{captions}>? Please output the relevant score directly. For example: 0.7256."""},
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
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                probs = extract_numbers(output_text[0])
                predict_scores  = probs[0]
                predict.append(predict_scores)
                max_index = predict.index(max(predict))
        try:
            messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text":'0'},
                                {"type": "image", "image": image_PIL[0]},
                                {"type": "text", "text":'1'},
                                {"type": "image", "image": image_PIL[1]},
                                {"type": "text", "text":'2'},
                                {"type": "image", "image": image_PIL[2]},
                                {"type": "text", "text":'3'},
                                {"type": "image", "image": image_PIL[3]},
                                {"type": "text", "text":'4'},
                                {"type": "image", "image": image_PIL[4]},
                                {"type": "text", "text": f'In the previous selection, image {max_index} was the most matching image. But is this really the picture that matches the caption best?'},
                                {"type": "text", "text": f'So,thinking carefully about the relationship between the caption and the images, carefully compare each image and answer the question.'},
                                {"type": "text", "text": f"""Which image is the most relevant one with the caption: <{captions}>? Please output relevant score for each image in json format. Every images should have a score The 0 index is the first image, 1 index is the second image, and so on. The score should be a float between 0 and 1."""},
                                {"type": "text", "text": """
                                For example:{{
                                    '0': 0.1234,
                                    '1': 0.5678,    
                                    '2': 0.9101,
                                    '3': 0.1121,
                                    '4': 0.3141
                                    }}"""},
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
            generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        top_k=50,
                        top_p=0.95,
                        repetition_penalty=1,
                    )
            generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
            output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
            probs = extract_json_from_string(output_text[0])
            sec_scores = []
            for key in probs.keys():
                sec_scores.append(float(probs[key]))
            weighted_scores = []
            alpha = 0.7
            for i in range(len(predict)):
                weighted_scores.append(alpha*float(predict[i]) + (1-alpha)*float(sec_scores[i]))
            weighted_scores = torch.softmax((torch.tensor(weighted_scores)) / 0.1, dim=0).tolist()
            max_index = weighted_scores.index(max(weighted_scores))
        
        # max_index = predict.index(max(predict))
            return max_index
        except:
            max_index = predict.index(max(predict))
            return max_index
    except:
        print('error')
        return 0

def Qwen_fine_reasoner_Retrieval_scores_weighted(image_PIL: list, captions: list, I2T=False, T2I=False, strategy="max", alpha=0.5):
    prompt_templates = [
        ("Does the image strongly match the caption: <{caption}>? When you analyizing the images ,just focus on the object in the caption .Don't care about others .Please output a float relevant score directly. For example: 0.7256.", 1),
        ("Evaluate how well the caption: <{caption}> describes the image. When you analyizing the images ,just focus on the object in the caption .Don't care about others .Output a float relevant score between 0 and 1. For example: 0.7256.", 1),
        ("How relevant is the caption: <{caption}> to the image content? When you analyizing the images ,just focus on the object in the caption .Don't care about others .Please output a float relevant score directly. For example: 0.7256.", 1),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predict = []

    for img in image_PIL:
        prompt_scores = []

        for prompt_text, weight in prompt_templates:
            prompt_filled = prompt_text.format(caption=captions[0])  # 注意这里是 captions[0]

            with torch.no_grad():
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_filled},
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

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                numbers = extract_numbers(output_text[0])
                if numbers:
                    try:
                        score = float(numbers[0])
                        prompt_scores.append(score)
                    except:
                        continue  # 忽略解析失败的结果

        # 对当前图片的所有 prompt score 进行集成
        if not prompt_scores:
            predict.append(0.0)
            continue

        from statistics import mean, median

        if strategy == "max":
            score = max(prompt_scores)
        elif strategy == "mean":
            score = mean(prompt_scores)
        elif strategy == "median":
            score = median(prompt_scores)
        elif strategy == "weighted":
            score = alpha * max(prompt_scores) + (1 - alpha) * mean(prompt_scores)
        else:
            raise ValueError("Invalid strategy. Use 'max', 'mean', 'median', or 'weighted'.")

        predict.append(score)

    try:
        max_index = predict.index(max(predict))
        return max_index
    except:
        return 0

def find_descr(topk_label:list,label_d:dict):
    topk_label_d = []
    for i in range(len(topk_label)):
        if topk_label[i] in label_d.keys():
            topk_label_d.append(label_d[topk_label[i]])
        else:
            topk_label_d.append(topk_label[i])
    return topk_label_d

def Qwen_fine_reasoner(image_path: str, top_k_labels,is_saliency=False, **kwargs):
        
    label_index = [str(i+1) for i in range(len(top_k_labels))]
    choices_dict = {label_index[i]: top_k_labels[i] for i in range(len(top_k_labels))}
    if 'awt_caption' in kwargs:
        top_k_labels_d = find_descr(top_k_labels, kwargs['awt_caption'])
        choices = [f'{label_index[i]}: {top_k_labels[i]} Description :{top_k_labels_d[i]}\n' for i in range(len(top_k_labels))]
    else:
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
                            {"type": "text", "text":'Query image'},
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": template},
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
    except:
        return None
def Qwen_fine_reasoner_Retrieval_v5(image_PIL:list,captions:list,I2T =False,T2I = False):
    predict = []
    # IMG = Image.open(d_path)
    
    with torch.no_grad():
        messages = [
                [
                    {"role": "user", "content": [
                        {"type": "image", "image": image_PIL[0]},
                        {"type": "text", "text": f"""Does the caption: <{cap}> match the image? Please directly output the relevance score in a format of "Answer: score" Keep four decimal places"""},
                        # {"type": "text", "text": f"""Does the daily image match the caption: <{captions}>? Please directly output the relevance score in a format of "Answer: score" Keep four decimal places"""},
                    ]}
                ]
                for cap in captions
            ]
        texts, image_inputs = [], []
        for msg in messages:
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            texts.append(text)
            img_input, _ = process_vision_info(msg)
            image_inputs.append(img_input[0])
        

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=77,

        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        probs = match_answer(output_text)
        # probs = extract_numbers(output_text[0])
        # predict_scores  = probs
        print(probs)
        try:
            predict_scores=[int(float(prob) * 100) / 100 for prob in probs]
        except:
            predict_scores = [0.0] * len(image_PIL)

        predict = predict_scores

    try:
        max_index = predict.index(max(predict))
        return max_index,predict
    except:
        return 0,None   
def Qwen_fine_reasoner_Retrieval_v6(image_PIL:list,captions:list,I2T =False,T2I = False):
    if I2T:
        with torch.no_grad():
            label_index = [str(i+1) for i in range(len(captions))]
            
            choices = [f'{label_index[i]}: {captions[i]}' for i in range(len(captions))]
            choices_dict = {label_index[i]: captions[i] for i in range(len(captions))}
            choices = ', '.join(choices)
            instruction = (
                            f"""You are given an image and 5 captions{choices}. Your task is to evaluate each caption in detail based on the following dimensions:

                            1. **Semantic Consistency**: How well does the content of the caption match the visual elements present in the image? Does it accurately describe what is happening or shown?
                            2. **Attention to Detail**: Does the caption capture specific details (e.g., objects, colors, actions, settings) accurately and vividly?
                            3. **Aesthetics**: Is the caption stylistically pleasing? Consider grammar, tone, fluency, and overall readability.
                            4. **Authenticity**: Does the caption feel genuine and plausible for the scene depicted? Avoid over-interpretation or assumptions not supported by the image.
                            5. **Other Relevant Factors** (such as clarity, relevance, emotional resonance, cultural appropriateness, etc.): Any additional aspects that contribute to the quality of the caption.

                            For each dimension, assign a score between 1 and 10 for each caption.Don't forget to analyze caption 0

                            After evaluating all captions across all dimensions, compute a total score for each caption by summing up its individual dimension scores.

                            Use a step-by-step chain-of-thought reasoning process to justify your evaluations. Clearly show your evaluation breakdown using the format below.

                            Please wrap your entire thought process inside `</think>` and `</think>` tags, and conclude with a final answer strictly in the format:
                            <answer>Answer:[X]</answer>

                            Example output format:

                            </think>
                            1. Semantic consistency: Caption 0 (9/10), Caption 1 (7/10), Caption 2 (9/10), Caption 3 (7/10), Caption 4 (7/10)
                            2. Attention to detail: Caption 0 (8/10), Caption 1 (6/10), Caption 2 (9/10), Caption 3 (5/10), Caption 4 (8/10)
                            3. Aesthetics: Caption 0 (9/10), Caption 1 (7/10), Caption 2 (8/10), Caption 3 (6/10), Caption 4 (9/10)
                            4. Authenticity: Caption 0 (9/10), Caption 1 (6/10), Caption 2 (9/10), Caption 3 (7/10), Caption 4 (8/10)
                            5. Other relevant factors: Caption 0 (8/10), Caption 1 (5/10), Caption 2 (9/10), Caption 3 (6/10), Caption 4 (8/10)

                            Total score: Caption 0: 9+8+9+9+8=43 | Caption 1: 7+6+7+6+5=31 | Caption 2: 9+9+8+9+9=44 | Caption 3: 7+5+6+7+6=31 | Caption 4: 7+8+9+8+8=40

                            <answer>Answer: 2</answer>
                            """

            )
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_PIL[0]},
                            {"type": "text", "text": instruction},
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

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,

                repetition_penalty=1.2
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            result = output_text[0].strip().replace('Output: ', '')
            reranked = match_answer(result)
            if reranked is None or reranked == {}:
                return None
            elif reranked is not None:
                first_key = int(reranked)
                return first_key

def Qwen_fine_reasoner_Retrieval_v2(image_PIL:list,captions:list,I2T =False,T2I = False,is_saliency=False):
    if I2T:

        with torch.no_grad():
            instruction = (
                        f"""What is in this picture?"""
                    )
            label_index = [str(i+1) for i in range(len(captions))]
            
            choices = [f'{label_index[i]}: {captions[i]}' for i in range(len(captions))]
            choices_dict = {label_index[i]: captions[i] for i in range(len(captions))}
            choices = ', '.join(choices)
            
            
            prompts = f"""Choose the most relevant sentence of the image from the following options: {choices}. You should directly output the answer index, for example: 1."""

            # prompts = f"""Choose the most relevant caption of the image from the following options: {choices}. Please directly output the  answer index in a format of "Answer: index" """

            template = instruction + "\n" + prompts 

            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_PIL[0]},
                            # {"type": "text", "text": template},
                            {"type": "text", "text": template},
                        ],
                    }
                ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True) 
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
            #     saliency_map = get_saliency_map(image_PIL[0], H_visual, W_visual)
            #     saliency_map = saliency_map.squeeze(0).squeeze(0).to(device) # [H, W]
            #     model.model.visual.saliency_embeds = saliency_map.unsqueeze(-1)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=77,
            )
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


    elif T2I:    
        with torch.no_grad():
            label_index = [str(i+1) for i in range(len(image_PIL))]
            
            choices = [f'{label_index[i]}: {image_PIL[i]}' for i in range(len(image_PIL))]
            choices_dict = {label_index[i]: image_PIL[i] for i in range(len(image_PIL))}
            test_prompt=  """Think carefully about the relationship between the caption and the images, and make a decision.
                        You should output the strongest image index.
                        First: analyze the caption and the images, then think about the relationship between them.
                        Second: choose the most relevant image.
                        Finally: output the answer index.
                        Let's think it through step by step."""
                        
                        
            prompts = 'Which image is most relevant to the caption?'
            instruction = (
                        # f"""
                        # Based on the given caption: {captions} .Choose the most relevant image .You should directly output the strongest image index, for example: 1.

                        # """
                        f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""
                    )
            template = prompts + "\n" + instruction

            messages = [
                    {
                        "role": "user",
                        "content": [

                        ],
                    }
                ]
            for img in image_PIL:
                messages[0]['content'].append(
                    {"type": "image", "image": img}
                )
            messages[0]['content'].append(
                    {"type": "text", "text": template}
            )
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
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                top_k=50,
                top_p=0.95,
                num_return_sequences=2,
                repetition_penalty=1.2
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            predict_img_idxs = []
            # print(output_text)
            for output in output_text:
                output = extract_numbers(output.strip().split('\n')[-1])
                if len(output) > 0:
                    if output[0] in list(choices_dict.keys()):
                        predict_img_idxs.append(output[0])
            if len(predict_img_idxs) > 0:
                predict_img_idx = max(set(predict_img_idxs), key=predict_img_idxs.count)
            else:
                predict_img_idx = None
            return predict_img_idx
        
def Qwen_fine_reasoner_Retrieval_v3(image_PIL:list,captions:list,I2T =False,T2I = False):
    if I2T:
        with torch.no_grad():
            instruction = (
                        f"""What is in this picture.
                            """
                    )
            label_index = [str(i+1) for i in range(len(captions))]
            
            choices = [f'{label_index[i]}: {captions[i]}' for i in range(len(captions))]
            choices_dict = {label_index[i]: captions[i] for i in range(len(captions))}
            choices = ', '.join(choices)
            # prompts = f"""Choose the most relevant description of the image from the following options: {choices}. You should directly output the answer index, for example: 1.
            #             First:analyze the caption and the images, then think about the relationship between them.
            #             Second: choose the most relevant caption.
            #             Finally: output the answer index.
            #             Let's think it through step by step."""
            prompts = f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""

            template = instruction + "\n" + prompts
            # print(template)
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_PIL[0]},
                            {"type": "text", "text": template},
                        ],
                    }
                ]
            messages = [messages for _ in range(5)]
            text = [processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=77,
                do_sample=True,
                temperature=0.95,
                top_k=75,
                top_p=0.8,
                repetition_penalty=1.2,
            )
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


    elif T2I:    
        with torch.no_grad():
            label_index = [str(i+1) for i in range(len(image_PIL))]
            
            choices = [f'{label_index[i]}: {image_PIL[i]}' for i in range(len(image_PIL))]
            choices_dict = {label_index[i]: image_PIL[i] for i in range(len(image_PIL))}
            prompts = 'Which image is most relevant to the caption?'
            # instruction = (
            #             f"""
            #             You are a visual reasoning model.
            #             Based on the given caption: {captions} .Choose the most relevant image .You should directly output the strongest image index, for example: 1.
            #             Think carefully about the relationship between the caption and the images, and make a decision.
            #             You should output the strongest image index.
            #             First: analyze the caption and the images, then think about the relationship between them.
            #             Second: choose the most relevant image.
            #             Finally: output the answer index.
            #             Let's think it through step by step.
            #             """
            #         )
            instruction = (
                        # f"""
                        # Based on the given caption: {captions} .Choose the most relevant image .You should directly output the strongest image index, for example: 1.

                        # """
                        f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""
                    )
            template = prompts + "\n" + instruction


            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f'{prompts}\n'},
                            {"type": "text", "text": f'1: '},
                            {"type": "text", "text": f'caption: The image shows two individuals sitting in shallow water, likely at a beach or a similar outdoor setting. The person on the left appears to be a woman wearing sunglasses and a colorful, patterned top with a red and white design. She has tattoos on her legs and is holding hands with the person on the right. The individual on the right seems to be a man who is shirtless, wearing black shorts, and also has tattoos visible on his legs. Both appear to be enjoying themselves, possibly engaging in a playful activity. In the background, there is a bottle of Coca-Cola placed on the sand, suggesting they might have been drinking it. The overall atmosphere of the image conveys a relaxed and fun moment outdoors.'},
                            {"type": "image", "image": image_PIL[0]},
                            {"type": "text", "text": f'2: '},
                            {"type": "text", "text": f"""The image captures a peaceful coastal scene on a sunny day, where three individuals are engaged in fishing activities along a rocky shoreline. One person, wearing a light blue sleeveless top, white cap, and light-colored shorts, stands holding a fishing rod, actively fishing. To their left, another individual dressed in a dark blue outfit with a white hat is also holding a fishing rod, seemingly assisting or participating. Further back, a third person in a black shirt and beige shorts stands near the water’s edge, possibly observing or waiting for a catch. The calm blue water gently laps against the rocks, with the distant horizon blending sea and sky, creating a sense of vastness and tranquility. The rocky shore, scattered with pebbles and stones, adds texture to the serene setting. With no other people or boats in sight, the location feels secluded and peaceful, evoking a deep connection with nature as the trio enjoys a quiet moment by the sea."""},
                            {"type": "image", "image": image_PIL[1]},
                            {"type": "text", "text": f'3: '},
                            {"type": "text", "text": f'caption:The image shows two individuals, an adult and a child, standing on a rocky shoreline near a body of water. The adult is wearing a red shirt, blue jeans, and a cap, while the child is dressed in a light-colored outfit with a green life vest. They appear to be looking at or interacting with something in the shallow water near the rocks. The setting suggests a calm, sunny day by the water, possibly engaging in a leisurely activity such as fishing or exploring the shore. '},
                            {"type": "image", "image": image_PIL[2]},
                            {"type": "text", "text": f'4: '},
                            {"type": "text", "text": f'caption:The image shows two silhouetted figures standing near the edge of a body of water during what appears to be either sunrise or sunset, as indicated by the warm, pinkish hues reflecting on the waters surface. The figures are wearing similar red tops and dark pants, suggesting they might be dressed for a casual outing or activity. The scene is serene, with gentle ripples in the water and a calm atmosphere. The lighting creates a dramatic contrast between the figures and the brightly lit background, emphasizing their outlines against the soft glow of the sky and water. '},
                            {"type": "image", "image": image_PIL[3]},
                            {"type": "text", "text": f'5: '},
                            {"type": "text", "text": f'caption: The image shows two children playing on a sandy beach. The child on the left is wearing a dark blue jacket, red pants, and appears to be walking or running. The child on the right is wearing a bright red shirt and shorts, and seems to be running or jumping with joy. The background features a sandy area with some grassy patches and dunes, suggesting a natural coastal environment. The lighting indicates it might be a sunny day.'},
                            {"type": "image", "image": image_PIL[4]},
                            {"type": "text", "text": f"""\nBased on the given caption: {captions}. Choose the most relevant image. You should directly output the image index, for example: 1."""},
                        ],
                    }
                ]
            messages = [messages for _ in range(5)]
            text = [processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=77,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            predict_img_idxs = []
            # print(output_text)
            for output in output_text:
                output = extract_numbers(output.strip().split('\n')[-1])
                if len(output) > 0:
                    if output[0] in list(choices_dict.keys()):
                        predict_img_idxs.append(output[0])
            if len(predict_img_idxs) > 0:
                predict_img_idx = max(set(predict_img_idxs), key=predict_img_idxs.count)
            else:
                predict_img_idx = None
            return predict_img_idx
        
def Qwen_fine_reasoner_Retrieval_v4(image_PIL:list,captions:list,I2T =False,T2I = False):
    if I2T:
        with torch.no_grad():
            instruction = (
                        f"""What is in this picture.
                            """
                    )
            label_index = [str(i+1) for i in range(len(captions))]
            
            choices = [f'{label_index[i]}: {captions[i]}' for i in range(len(captions))]
            choices_dict = {label_index[i]: captions[i] for i in range(len(captions))}
            choices = ', '.join(choices)
            # prompts = f"""Choose the most relevant description of the image from the following options: {choices}. You should directly output the answer index, for example: 1.
            #             First:analyze the caption and the images, then think about the relationship between them.
            #             Second: choose the most relevant caption.
            #             Finally: output the answer index.
            #             Let's think it through step by step."""
            prompts = f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""

            template = instruction + "\n" + prompts
            # print(template)
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_PIL[0]},
                            {"type": "text", "text": template},
                        ],
                    }
                ]
            messages = [messages for _ in range(5)]
            text = [processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=77,
                do_sample=True,
                temperature=0.95,
                top_k=75,
                top_p=0.8,
                repetition_penalty=1.2,
            )
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


    elif T2I:    
        with torch.no_grad():
            label_index = [str(i+1) for i in range(len(image_PIL))]
            
            choices = [f'{label_index[i]}: {image_PIL[i]}' for i in range(len(image_PIL))]
            choices_dict = {label_index[i]: image_PIL[i] for i in range(len(image_PIL))}
            prompts = 'Which image is most relevant to the caption?'
            # instruction = (
            #             f"""
            #             You are a visual reasoning model.
            #             Based on the given caption: {captions} .Choose the most relevant image .You should directly output the strongest image index, for example: 1.
            #             Think carefully about the relationship between the caption and the images, and make a decision.
            #             You should output the strongest image index.
            #             First: analyze the caption and the images, then think about the relationship between them.
            #             Second: choose the most relevant image.
            #             Finally: output the answer index.
            #             Let's think it through step by step.
            #             """
            #         )
            instruction = (
                        # f"""
                        # Based on the given caption: {captions} .Choose the most relevant image .You should directly output the strongest image index, for example: 1.

                        # """
                        f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""
                    )
            template = prompts + "\n" + instruction


            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_PIL[0]},
                            {"type": "image", "image": image_PIL[1]},
                            {"type": "image", "image": image_PIL[2]},
                            {"type": "image", "image": image_PIL[3]},
                            {"type": "image", "image": image_PIL[4]},
                            {"type": "text", "text": f"""Which image is the most relevant one with the caption: <{captions}>? Step 1: describe each image in detail.\n Step 2: choose the most relevant image.\n Step 3: Directly give the answer index, for example: 1. Let's think it through step by step."""},
                        ],
                    }
                ]
            messages = [messages for _ in range(5)]
            text = [processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            predict_img_idxs = []
            print(output_text)
            for output in output_text:
                output = extract_numbers(output.strip().split('\n')[-1])
                if len(output) > 0:
                    if output[0] in list(choices_dict.keys()):
                        predict_img_idxs.append(output[0])
            if len(predict_img_idxs) > 0:
                predict_img_idx = max(set(predict_img_idxs), key=predict_img_idxs.count)
            else:
                predict_img_idx = None
            return predict_img_idx
        
def Qwen_fine_reasoner_Retrieval_caption(image_PIL:list,captions:list, descrption: list, I2T =False,T2I = False):
    if I2T:
        with torch.no_grad():
            instruction = (
                        f"""What is in this picture.
                            """
                    )
            label_index = [str(i+1) for i in range(len(captions))]
            
            choices = [f'{label_index[i]}: {captions[i]}' for i in range(len(captions))]
            choices_dict = {label_index[i]: captions[i] for i in range(len(captions))}
            choices = ', '.join(choices)
            # prompts = f"""Choose the most relevant description of the image from the following options: {choices}. You should directly output the answer index, for example: 1.
            #             First:analyze the caption and the images, then think about the relationship between them.
            #             Second: choose the most relevant caption.
            #             Finally: output the answer index.
            #             Let's think it through step by step."""
            prompts = f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""

            template = instruction + "\n" + prompts
            # print(template)
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_PIL[0]},
                            {"type": "text", "text": template},
                        ],
                    }
                ]
            messages = [messages for _ in range(5)]
            text = [processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=77,
                do_sample=True,
                temperature=0.95,
                top_k=75,
                top_p=0.8,
                repetition_penalty=1.2,
            )
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


    elif T2I:    
        with torch.no_grad():
            label_index = [str(i+1) for i in range(len(descrption))]
            
            choices = [f'{label_index[i]}: {descrption[i][0]}\n\n' for i in range(len(descrption))]
            choices_dict = {label_index[i]: descrption[i][0] for i in range(len(descrption))}
            prompts = 'Which image is most relevant to the caption?'
            choices = ''.join(choices)
            # instruction = (
            #             f"""
            #             You are a visual reasoning model.
            #             Based on the given caption: {captions} .Choose the most relevant image .You should directly output the strongest image index, for example: 1.
            #             Think carefully about the relationship between the caption and the images, and make a decision.
            #             You should output the strongest image index.
            #             First: analyze the caption and the images, then think about the relationship between them.
            #             Second: choose the most relevant image.
            #             Finally: output the answer index.
            #             Let's think it through step by step.
            #             """
            #         )
            instruction = (
                        # f"""
                        # Based on the given caption: {captions} .Choose the most relevant image .You should directly output the strongest image index, for example: 1.

                        # """
                        f"""Given above five descriptions, Which sentence is the most relevant one with the caption: <{captions}>. You should directly output the description index, for example: 1."""
                    )
            # template = prompts + "\n" + instruction


            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f'{choices}'},
                            {"type": "text", "text": f"""{instruction}"""},
                        ],
                    }
                ]
            print(messages)
            messages = [messages for _ in range(5)]
            text = [processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=77,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            predict_img_idxs = []
            print(output_text)
            for output in output_text:
                output = extract_numbers(output.strip().split('\n')[-1])
                if len(output) > 0:
                    if output[0] in list(choices_dict.keys()):
                        predict_img_idxs.append(output[0])
            if len(predict_img_idxs) > 0:
                predict_img_idx = max(set(predict_img_idxs), key=predict_img_idxs.count)
            else:
                predict_img_idx = None
            return predict_img_idx
def Qwen_fine_reasoner_Retrieval_Interleaved(image_PIL:list,captions:list, topk_caption: list, I2T =False,T2I = False):
    if I2T:
        with torch.no_grad():
            instruction = (
                        f"""What is in this picture.
                            """
                    )
            label_index = [str(i+1) for i in range(len(captions))]
            
            choices = [f'{label_index[i]}: {captions[i]}' for i in range(len(captions))]
            choices_dict = {label_index[i]: captions[i] for i in range(len(captions))}
            choices = ', '.join(choices)
            # prompts = f"""Choose the most relevant description of the image from the following options: {choices}. You should directly output the answer index, for example: 1.
            #             First:analyze the caption and the images, then think about the relationship between them.
            #             Second: choose the most relevant caption.
            #             Finally: output the answer index.
            #             Let's think it through step by step."""
            prompts = f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""

            template = instruction + "\n" + prompts
            # print(template)
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_PIL[0]},
                            {"type": "text", "text": template},
                        ],
                    }
                ]
            messages = [messages for _ in range(5)]
            text = [processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=77,
                do_sample=True,
                temperature=0.95,
                top_k=75,
                top_p=0.8,
                repetition_penalty=1.2,
            )
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


    elif T2I:    
        with torch.no_grad():
            label_index = [str(i+1) for i in range(len(image_PIL))]
            
            choices = [f'{label_index[i]}: {image_PIL[i]}' for i in range(len(image_PIL))]
            choices_dict = {label_index[i]: image_PIL[i] for i in range(len(image_PIL))}
            prompts = 'Which image is most relevant to the caption?'
            choices = ''.join(choices)
            instruction = (

                        f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""
                    
                    )
            # template = prompts + "\n" + instruction


            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f'{instruction}\n'},
                            {"type": "text", "text": f'1: '},
                            {"type": "text", "text": f'{topk_caption[0][0]}'},
                            {"type": "image", "image": image_PIL[0]},
                            {"type": "text", "text": f'2: '},
                            {"type": "text", "text": f'{topk_caption[1][0]}'},
                            {"type": "image", "image": image_PIL[1]},
                            {"type": "text", "text": f'3: '},
                            {"type": "text", "text": f'{topk_caption[2][0]}'},
                            {"type": "image", "image": image_PIL[2]},
                            {"type": "text", "text": f'4: '},
                            {"type": "text", "text": f'{topk_caption[3][0]}'},
                            {"type": "image", "image": image_PIL[3]},
                            {"type": "text", "text": f'5: '},
                           {"type": "text", "text": f'{topk_caption[4][0]}'},
                            {"type": "image", "image": image_PIL[4]},
                            {"type": "text", "text": f"""\nBased on the given caption: {captions}. Choose the most relevant image. You should directly output the image index, for example: 1."""},
                        ],
                    }
                ]


            messages = [messages for _ in range(5)]
            text = [processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=77,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            predict_img_idxs = []
            # print(output_text)s
            for output in output_text:
                output = extract_numbers(output.strip().split('\n')[-1])
                if len(output) > 0:
                    if output[0] in list(choices_dict.keys()):
                        predict_img_idxs.append(output[0])
            if len(predict_img_idxs) > 0:
                predict_img_idx = max(set(predict_img_idxs), key=predict_img_idxs.count)
            else:
                predict_img_idx = None
            return predict_img_idx

def Qwen_fine_reasoner_Retrieval(image_PIL:list,captions:list,I2T =False,T2I = False):
    if I2T:
        with torch.no_grad():
            instruction = (
                        f"""Identify the strongest caption in the given image from the options: {captions}.
                            Answer which caption is most relevant to the image.
                            Output the strongest caption in json format only.
                            Do not include weaker captions in the output.
                            """
                    )

                # 提供多个三位小数非零的示例
            example = (
                    "Examples:"
                    
                    "1. If the captions are: 'Cat', 'Dog', 'Bird', 'Fish'"
                    "Input: <image of a dog>"
                    "Output:"
                        '{"Dog": 0.723}'
                    "2. If the captions are: 'car', 'bike', 'truck'"
                    "Input: <image of a car>"
                    "Output:"
                       '{"Car": 0.723}'

                    "3. If the captions are: 'elephant', 'lion', 'zebra', 'giraffe'"
                    "Input: <image of an elephant>"
                    "Output:"
                        '{"Elephant": 0.723}'

                )
            template = instruction + "\n" + example

            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_PIL[0]},
                            {"type": "text", "text": template},
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

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                repetition_penalty=1.2
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            result = output_text[0].strip().replace('Output: ', '')
            reranked = extract_json_from_string(result)
            if reranked is None or reranked == {}:
                return None
            elif reranked is not None:
                first_key = next(iter(reranked))
                return first_key
    elif T2I:    
        with torch.no_grad():
            instruction = (
                        f"""Based on the given caption: {captions[0]}, identify the most relevant image.
                            Answer which image is most relevant to the caption.
                            Output the strongest photo index in json format.
                            """
                    )

                # 提供多个三位小数非零的示例
            example = (
                    "Examples:"
                    
                    "1. If the pictures are: 'image1', 'image2', 'image3', 'image4'"
                    "Input: a photo of a dog"
                    "Output:"
                        "{'1':0.99}"

                    "2. If the pictures are: 'image1', 'image2', 'image3', 'image4'"
                    "Input: a photo of a car"
                    "Output:"
                       "{'0':0.99}"

                    "3. If the pictures are: 'image1', 'image2', 'image3', 'image4'"
                    "Input: a photo of a elephant"
                    "Output:"
                        "{'3':0.99}"

                )
            template = instruction + "\n" + example

            messages = [
                    {
                        "role": "user",
                        "content": [

                        ],
                    }
                ]
            for i in range(len(image_PIL)):
                messages[0]['content'].append(
                    {"type": "image", "image": image_PIL[i]}
                )
            messages[0]['content'].append(
                    {"type": "text", "text": template}
            )
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

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                repetition_penalty=1.2
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            result = output_text[0].strip().replace('Output: ', '')
            reranked = extract_json_from_string(result)
            if reranked is None or reranked == {}:
                return None

            first_key = next(iter(reranked))
            return first_key
        
def get_caption(images, output_file = '/mnt/d/dws/MCS/Datasets/coco_caption.json'):
    caption = []
    for each_image in images:
        ## get caption from qwen
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": each_image},
                            {"type": "text", "text": f"""Please describe the image in detail."""},
                        ],
                    }
                ]
        messages = [messages for _ in range(5)]
        text = [processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            repetition_penalty=1.2
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        caption.append(output_text)
        with open(output_file, 'w') as f:
            json.dump(caption, f)

def get_caption_incontext(images, output_file = '/mnt/d/dws/MCS/Datasets/flickr_simple_caption.json'):
    caption = []
    nums = 0
    from tqdm  import tqdm
    bar = tqdm(total=len(images))
    for each_image in images:
        # print(nums)
        nums += 1
        ## get caption from qwen
        # messages = [
        #             {
        #                 "role": "user",
        #                 "content": [
        #                     {"type": "image", "image": images[0]},
        #                     {"type": "text", "text": f"""Caption: The image shows a man with pierced ears is wearing glasses and an orange hat. The man wears a gray shirt with black trim on their shoulders, standing indoors against a blurred background that suggests other people might be present but out of focus. The hat is  made from orange yarn or fabric adorned with printed labels resembling those found on packaged snacks like Ritz crackers. The design includes red text and white graphics typical for such packaging, creating a playful juxtaposition between food branding and headwear fashion."""},
        #                     {"type": "image", "image": images[5]},
        #                     {"type": "text", "text": f"""Caption: The image shows a man in light-colored clothing taking pictures of a group of men in dark suits and bowler hats who surround a woman in a white strapless dress. The photographer in light-colored clothing (beige shirt and khaki pants) stands by holding up his camera. The main focus is on a group of formally dressed men: a group of men in black suits with white shirts and ties, and a woman in a white gown sitting in the center of these men."""},
        #                     {"type": "image", "image": each_image},
        #                     {"type": "text", "text": f"""Caption: """},
        #                 ],
        #             }
        #         ]
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": each_image},
                            {"type": "text", "text": f"""Describe thie image """},
                        ],
                    }
                ]
        messages = [messages for _ in range(5)]
        text = [processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            repetition_penalty=1.2
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        caption.append(output_text[0])
        with open(output_file, 'w') as f:
            json.dump(caption, f)
        bar.update(1)
    bar.close()
    
def Qwen_fine_reasoner_Retrieval_scores_v2(scores,image_PIL:list,captions:list, I2T =False,T2I = False):
    scores = scores.tolist()
    label_index = [str(i+1) for i in range(len(image_PIL))]
    choices = [f'{label_index[i]}: {image_PIL[i]}' for i in range(len(image_PIL))]
    choices = ''.join(choices)
    predict = []

    for img in image_PIL:
        with torch.no_grad():
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": f"""Does the image match the caption: <{captions}>? Please output the relevant score directly. For example: 0.7256."""},
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
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            probs = extract_numbers(output_text[0])
            predict_scores  = probs[0]
            predict.append(predict_scores)
            weighted_scores = []
            alpha = 0.9
            for i in range(len(predict)):
                weighted_scores.append(alpha*float(predict[i]) + (1-alpha)*float(scores[i]))
    top2=[]
    max_index = weighted_scores.index(max(weighted_scores))
    top2.append(max_index)
    second_max = weighted_scores.index(sorted(weighted_scores)[-2])
    top2.append(second_max)
    if weighted_scores[max_index]-weighted_scores[second_max]<0.1:
        try:

            messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"""0"""},
                                {"type": "image", "image": image_PIL[max_index]},
                                {"type": "text", "text": f"""1"""},
                                {"type": "image", "image": image_PIL[second_max]},
                                {"type": "text", "text": f"""Which image is the most relevant one with the caption: {captions} You should directly output the image index, for example: 1."""},
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
            generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1,
                )
            generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
            output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            probs = extract_numbers(output_text[0])
            predict_index = int(probs[0])
            predict_index = top2[predict_index]
            return str(predict_index)
        except:
            return 0
    else:
        return str(max_index)

def Qwen_fine_reasoner_Retrieval_scores_SAM(image_PIL:list,captions, I2T =False,T2I = False):
    
    predict = []
    # try:
    #     # seg_images, boxed_images = process_imglist(image_list=image_PIL,captions=captions)
    # except:
    #     seg_images = [img for img in image_PIL]
    #     print('segmentation failed')
    seg_images = [img for img in image_PIL]
    for img,seg_img in zip(image_PIL,seg_images):
        with torch.no_grad():
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "image", "image": seg_img},
                            #{"type": "text", "text": f"""Dose the image strongly match the caption:{captions}? Please perform the task by first stating your intent as a high-level plan for approaching the problem or task. Then, provide the step-by-step reasoning or generation process, ensuring it aligns with the stated intent .You should also analyze the weaknesses  Please output the relevance score in a format of "Answer: score" Keep twe decimal places"""},
                            {"type": "text", "text": f"""Does the image match the caption: <{captions}>?  Please directly output the relevance score in a format of "Answer: score".Keep four decimal places"""},
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
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            predict_scores = match_answer(output_text[0])
            predict.append(predict_scores)
    try:
        max_index = predict.index(max(predict))
        return max_index,predict
    except:
        return 0
def select_or_fuse(scores_A, scores_B, epsilon =0.03,temperature=0.7):
    zero = torch.tensor([0.0 for _ in range(len(scores_A))], dtype=torch.float16)
    if scores_A is None:
        scores_A = zero
    if scores_B is None:
        scores_B = zero
    if scores_A is not None and scores_B is not None:
        max_1 = max(scores_A)
        second_1 = sorted(scores_A)[-2]
        max_2 = max(scores_B)
        second_2 = sorted(scores_B)[-2]
        scores_A = torch.tensor(scores_A, dtype=torch.float16)
        scores_B = torch.tensor(scores_B, dtype=torch.float16)
        delta_a = max_1 - second_1 if max_1 != second_1 else max_1-sorted(scores_A)[-3]
        delta_b = max_2 - second_2 if max_2 != second_2 else max_2-sorted(scores_B)[-3]
        if delta_a > delta_b+epsilon:
            return scores_A
        elif delta_b > delta_a+epsilon:
            return scores_B
        else:
            soft_A = torch.softmax(scores_A / temperature, dim=0)
            soft_B = torch.softmax(scores_B / temperature, dim=0)
            total_delta = delta_a + delta_b if (delta_a + delta_b) > 0 else 1e-6
            alpha = delta_a / total_delta   
            final=alpha * soft_A + (1 - alpha) * soft_B
            return final


def Qwen_fine_reasoner_Retrieval_setwise(image_PIL:list,captions:list, I2T =False,T2I = False):
    try:
        window_size = 3
        step = 3
        def sliding_window(seq, window_size, step=1):
            result = []
            for i in range(0, len(seq) - window_size + 1, step):
                result.append(seq[i:i + window_size])
            return result
        set = sliding_window(image_PIL,window_size,step)
        
        sub_bests = []
        for sub_set in set:
            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        # {"type": "text", "text":f'Image index:{image_PIL.index(sub_set[0])}'},
                                        {"type": "image", "image": sub_set[0]},
                                        # {"type": "text", "text":f'Image index:{image_PIL.index(sub_set[1])}'},
                                        {"type": "image", "image": sub_set[1]},
                                        # {"type": "text", "text":f'Image index:{image_PIL.index(sub_set[2])}'},
                                        {"type": "image", "image": sub_set[2]},
                                        # {"type": "text", "text": f"""
                                        # You need to analyze the images and the caption step by step
                                        # Every image has an index 
                                        # The first image index is{image_PIL.index(sub_set[0])}
                                        # The second image index is{image_PIL.index(sub_set[1])}
                                        # The third image index is{image_PIL.index(sub_set[2])}
                                        # Which image is  match the caption:{captions}? Please directly output the image index in a format of "Answer: image_index"  and use no more than 60 words to explain why.for example: 'Answer: 8' """},
                                        {"type": "text", "text": f"""
                                        You need to analyze the images and the caption step by step
                                        Every image has an index 
                                        The first image index is{image_PIL.index(sub_set[0])}
                                        The second image index is{image_PIL.index(sub_set[1])}
                                        The third image index is{image_PIL.index(sub_set[2])}
                                        Which image is  match the caption:{captions}? Please directly output the image index in a format of "Answer: index" .for example: 'Answer: 8' """},
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
            generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=77,
                    repetition_penalty=1.2
                )
            generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
            output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            # print(output_text)
            sub_best = match_answer(output_text[0])
            sub_bests.append((int(sub_best) if type(sub_best)==float else None))
            
        get_non_none_element = lambda lst: next((x for x in lst if x is not None), None)

        if sub_bests[0]==sub_bests[1]: return int(sub_bests[0])
        if sub_bests[0]==None or sub_bests[1]==None: return get_non_none_element(sub_bests)
        
        
        messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        # {"type": "text", "text":f'Image index:{sub_bests[0]}'},
                                        {"type": "image", "image": image_PIL[sub_bests[0]]},
                                        # {"type": "text", "text":f'Image index:{sub_bests[1]}'},
                                        {"type": "image", "image": image_PIL[sub_bests[1]]},
                                        # {"type": "text", "text": f"""You need to analyze the images and the caption step by step
                                        # Every image has an index 
                                        # The first image index is {sub_bests[0]}
                                        # The second image index is {sub_bests[1]}
                                        # Which image is  match the caption:{captions}? Please directly output the image index in a format of "Answer: image_index"  and use no more than 60 words to explain why .for example: 'Answer: 8' """},
                                        {"type": "text", "text": f"""
                                        You need to analyze the images and the caption step by step
                                        Every image has an index 
                                        The first image index is{image_PIL.index(sub_set[0])}
                                        The second image index is{image_PIL.index(sub_set[1])}
                                        The third image index is{image_PIL.index(sub_set[2])}
                                        Which image is  match the caption:{captions}? Please directly output the image index in a format of "Answer: index" .for example: 'Answer: 8' """},                                        
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
        generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=77,
                        repetition_penalty=1.2
                    )
        generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
        output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
        # print(output_text)
        predict_index = match_answer(output_text[0])
        if int(predict_index) not in sub_bests:
            if int(predict_index)==1:
                return int(sub_bests[0])
            elif int(predict_index)==2:
                return int(sub_bests[1])
        return int(predict_index)
    except:
        return 0

    
def Qwen_fine_reasoner_Retrieval_log_setwise(image_PIL: list, captions: list, I2T=False, T2I=False):
    sub_bests = []
    num_ids = [tokenizer.encode(f"{i}", add_special_tokens=False)[0] for i in range(5)]
    window_size = 3
    step = 2
    def sliding_window(seq, window_size, step=1):
        result = []
        for i in range(0, len(seq) - window_size + 1, step):
            result.append(seq[i:i + window_size])
        return result
    set = sliding_window(image_PIL,window_size,step)
    for sub_set in set:
        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": sub_set[0]},
                                    {"type": "image", "image": sub_set[1]},
                                    {"type": "image", "image": sub_set[2]},
                                    {"type": "text", "text": f'The fisrt image index is {image_PIL.index(sub_set[0])}'},
                                    {"type": "text", "text": f'The second image index is {image_PIL.index(sub_set[1])}'},
                                    {"type": "text", "text": f'The third image index is {image_PIL.index(sub_set[2])}'},
                                    {"type": "text", "text": f"""Which image is strongest match the caption{captions}? Please directly output the image index in a format of "Answer: index"  for example: 'Answer: 1' """},
                                ],
                            }
                        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=text,images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt").to(device)
        
        start_index = torch.where(inputs['input_ids'][0] == 1922)[0][-1] + 1
        temp = inputs['input_ids'][0][-5:] 
        temp1 = inputs['input_ids'][0][start_index:-5]  
        inputs['input_ids'][0][start_index:] = torch.cat([temp, temp1], dim=-1) 
        
        generated_ids = model(**inputs,)
        logits = generated_ids['logits'][0][(start_index+4):-1]  ### 15, voc_size
        logits = torch.softmax(logits,dim=-1)
        
        ans =logits[0]
        num_ids = [tokenizer.encode(f"{i}", add_special_tokens=False)[0] for i in [image_PIL.index(sub_set[0]),image_PIL.index(sub_set[1]),image_PIL.index(sub_set[2])]]
        prob_list = torch.tensor([ans[id] for id in num_ids])
        predict_index = processor.decode(num_ids[torch.argmax(prob_list).tolist()])
        sub_bests.append(int(predict_index))
    if sub_bests[0]==sub_bests[1]: return int(sub_bests[0])
    messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": image_PIL[sub_bests[0]]},
                                        {"type": "image", "image": image_PIL[sub_bests[1]]},
                                        {"type": "text", "text": f'The fisrt image index is {sub_bests[0]}'},
                                        {"type": "text", "text": f'The second image index is {sub_bests[1]}'},
                                        {"type": "text", "text": f"""Which image is strongest match the caption{captions}? Please directly output the image index in a format of "Answer: index"  for example: 'Answer: 1' """},
                                    ],
                                }
                            ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text,images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt").to(device)
    
    start_index = torch.where(inputs['input_ids'][0] == 1922)[0][-1] + 1
    temp = inputs['input_ids'][0][-5:] 
    temp1 = inputs['input_ids'][0][start_index:-5]  
    inputs['input_ids'][0][start_index:] = torch.cat([temp, temp1], dim=-1) 
    
    generated_ids = model(**inputs,)
    
    logits = generated_ids['logits'][0][(start_index+4):-1]  ### 15, voc_size
    logits = torch.softmax(logits,dim=-1)
    
    num_ids = [tokenizer.encode(f"{i}", add_special_tokens=False)[0] for i in [sub_bests[0],sub_bests[1]]]
    prob_list = torch.tensor([logits[0][id] for id in num_ids])
    predict_index = processor.decode(num_ids[torch.argmax(prob_list).tolist()])

    
    return int(predict_index)

def Qwen_fine_compare(image_path: str, top_k_labels,is_saliency=True, **kwargs):
        
    label_index = [str(i+1) for i in range(len(top_k_labels))]
    choices_dict = {label_index[i]: top_k_labels[i] for i in range(len(top_k_labels))}

    choices = [f'{label_index[i]}: {top_k_labels[i]}\n' for i in range(len(top_k_labels))]

    try:
        with torch.no_grad():
            choices = ', '.join(choices)
            instruction = (            
                """What is in the picture"""
                # "What type of news is this image related to?"
            )
            prompts = f"""Choose the most relevant description from the following options : \n{choices}. You should directly output the answer index, for example: 1."""
            template = instruction + "\n" + prompts
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": template},
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
    except:
        return None

def Qwen_fine_reasoner_Retrieval_setwise_v2(image_PIL:list,captions:list, I2T =False,T2I = False):

    # try:
        window_size = 3
        step = 2
        def sliding_window(seq, window_size, step=1):
            result = []
            for i in range(0, len(seq) - window_size + 1, step):
                result.append(seq[i:i + window_size])
            return result
        set = sliding_window(image_PIL,window_size,step)
        
        sub_bests = []
        
        for i,sub_set in enumerate(set):
            if i!=0:
                sub_set[0] = image_PIL[sub_bests[0]]
            messages = [
                    {
                        "role": "system",
                        "content": """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant provides the user with the answer enclosed within <answer> </answer> tags, i.e., <answer> answer here </answer>.""",
                    },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text":'1'},
                                        {"type": "image", "image": sub_set[0]},
                                        {"type": "text", "text":'2'},
                                        {"type": "image", "image": sub_set[1]},
                                        {"type": "text", "text":'3'},
                                        {"type": "image", "image": sub_set[2]},
                                        {"type": "text", "text": f"""Which image match the caption:{captions} most? Let's think step by step"""},
                                        # {"type": "text", "text": f"""Please provide the index of the most relevant image to the query, enclosed in square brackets,within the answer tags.And explain why in a short sentence The index of an image is not the order of the image. For example, If the image with index [2] is the most relevant, the answer should be:<answer>[2]</answer>"""},
                                        {"type": "text", "text": f"""Please output your reasoning process in a short sentence within the think tags and provide the index of the most relevant image to the query, enclosed in square brackets,within the answer tags. For example, If the image with index [2] is the most relevant, the reasoning process should be <think>[REASIONG]</think>the answer should be:<answer>[2]</answer>"""},
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
            generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    repetition_penalty=1.2
                )
            generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
            output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            # print(output_text)
            sub_best = re.search(r'\d+', output_text[0].split('<answer>', 1)[1]).group()
            if int(sub_best)==0:sub_best = 1
            sub_best = image_PIL.index(sub_set[int(sub_best)-1])
            if i!=0:
                return int(sub_best)
            sub_bests.append(int(sub_best))
            

    # except:
    #     return 0

# fusion_mission = 'multi_text' # multi_image
# method = 'adapt' # default adapt

Multi_layers_fusion(
            self=model,
            starting_layer=5,
            ending_layer=16,
            entropy_threshold=0.75,
            mission = 'multi_text',#multi_image
            retracing_ratio=0.12,#0.05-0.35
            # retracing_ratio=0.12,#0.05-0.35
            vision_retracing_method = 'adapt',#default adapt
        )

path = '/root/dws/MCS/Datasets/MMEB/GQA/image_2328196.jpg'
# image_path = [os.path.join(path,f'{i}.jpg') for i in range(5]
image_path = [path,path,path,path,path]
image_PIL = []
for image in image_path:
    img = Image.open(image)
    image_PIL.append(img)
i =0 
captions = " A man with glasses is wearing a beer can crocheted hat."
import time
start_time = time.time()
predict_index = Qwen_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL,captions, is_saliency=True,I2T = False,T2I = True)
end_time = time.time()
print("Time taken:", end_time - start_time)
print(predict_index)


# image_path = '/root/dws/MCS/Datasets/MMEB/GQA/image_2328196.jpg'
# image_PIL = Image.open(image_path).convert('RGB')
# import time
# start_time = time.time()
# captions =   ['Man skates along cement wall.', '"A skateboarder rides up a concrete wall, nearly falling off as he tries a trick."', 'A skateboarder is riding his board along the boundary stone of a parking lot.', 'The man does a trick on his skateboard on a concrete ramp.', 'A man does skateboard tricks in a parking lot at night.']
# predict_index = Qwen_fine_reasoner_Retrieval_v5([image_PIL],captions,I2T = True,T2I = False)
# end_time = time.time()
# print("Time taken:", end_time - start_time)
# print(predict_index)


def demo_attention_visualization():
    """
    演示如何使用注意力可视化功能
    """
    # 示例：加载图像和设置captions
    # image_path = '/path/to/your/image.jpg'
    # image_PIL = [Image.open(image_path).convert('RGB')]
    # captions = ["your caption here", None]  # [text_caption, query_image_or_None]
    
    print("=== 注意力可视化演示 ===")
    print("使用方法：")
    print("1. 准备图像列表 image_PIL")
    print("2. 准备captions")
    print("3. 调用函数并启用注意力输出")
    print()
    
    example_code = '''
    # 示例代码
    from PIL import Image
    
    # 加载图像
    image_path = "/path/to/your/image.jpg"
    image_PIL = [(Image.open(image_path).convert('RGB'), "image_caption")]
    captions = ["your query text", None]  # [query_text, query_image_or_None]
    
    # 调用函数并启用注意力可视化
    result = Qwen_fine_reasoner_Retrieval_scores_entropy_saliency(
        image_PIL=image_PIL,
        captions=captions,
        output_attention=True,  # 启用注意力输出
        save_attention_path="/path/to/save/attention"  # 保存路径
    )
    
    # 解析结果
    if len(result) == 4:  # 包含注意力信息
        best_index, predict_scores, entropies, attention_maps = result
        print(f"最佳匹配索引: {best_index}")
        print(f"预测分数: {predict_scores}")
        print(f"熵值: {entropies}")
        print(f"注意力图数量: {len(attention_maps) if attention_maps else 0}")
        
        # 手动可视化特定图像的注意力
        if attention_maps and attention_maps[0] is not None:
            visualize_attention_on_image(
                image_PIL[0][0],  # 原始图像
                attention_maps[0],  # 对应的注意力图
                "/path/to/save/manual_attention.png"
            )
    else:
        best_index, predict_scores, entropies = result
        print("未启用注意力输出")
    '''
    
    print(example_code)
    print("\n=== 功能说明 ===")
    print("1. output_attention=True: 启用注意力权重输出")
    print("2. save_attention_path: 自动保存注意力可视化图像的路径前缀")
    print("3. 会生成两种可视化:")
    print("   - overlay图: 注意力热力图叠加在原图上")
    print("   - heatmap图: 纯注意力热力图")
    print("4. 返回值包含attention_maps，可用于进一步分析")
    print("\n=== 注意事项 ===")
    print("1. 确保安装了matplotlib和opencv-python")
    print("2. 注意力可视化需要额外的GPU内存")
    print("3. 视觉token位置的计算可能需要根据具体模型调整")

if __name__ == "__main__":
    # 取消注释下面这行来运行演示
    # demo_attention_visualization()
    pass