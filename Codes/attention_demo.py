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
from scipy.ndimage import zoom, gaussian_filter
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
import math
# from Grounded_sam2 import process_imglist
from PIL import Image
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
model_id = '/root/dws/MCS/Models/Qwen2.5-VL-7B-Instruct'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).to(device).eval()
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


def get_featuremap_size(image):

    width, height = image.size

    # Qwen2.5-VL使用14x14的patch尺寸
    patch_size = processor.image_processor.patch_size  # 14
    
    # 计算理论特征图的高度和宽度（token合并前）
    H_theoretical = height // patch_size
    W_theoretical = width // patch_size
    
    print(f"图像尺寸: {width} x {height}")
    print(f"Patch尺寸: {patch_size}")
    print(f"理论特征图尺寸: {H_theoretical} x {W_theoretical} = {H_theoretical * W_theoretical} tokens")
    
    # 返回理论值，实际的token数量会在后面根据真实数据确定
    return H_theoretical, W_theoretical


def get_vision_attention_map(attention_weights, vision_start, vision_end, H_theoretical, W_theoretical, actual_tokens=None, output_tokens_start=None):

    if output_tokens_start is not None:
        aggregated_attention = attention_weights[0, output_tokens_start, vision_start:vision_end]  # [num_vision_tokens]
        print(f"使用位置 {output_tokens_start} 的token对视觉区域的注意力")
    else:
        aggregated_attention = attention_weights[0, -1, vision_start:vision_end]  # [num_vision_tokens]
        print("使用最后一个token的注意力")

    attention_values = aggregated_attention.detach().cpu().numpy()
    
    # 检查是否有NaN值
    if np.isnan(attention_values).any():
        print("警告: 发现NaN值，将其替换为0")
        attention_values = np.nan_to_num(attention_values, nan=0.0)
    
    if actual_tokens:
        # 尝试找到接近理论宽高比的因式分解
        target_ratio = W_theoretical / H_theoretical
        best_h, best_w = 1, actual_tokens
        best_ratio_diff = float('inf')
        
        # 寻找所有可能的因式分解
        for h in range(1, int(actual_tokens**0.5) + 5):
            if actual_tokens % h == 0:
                w = actual_tokens // h
                ratio = w / h
                ratio_diff = abs(ratio - target_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_h, best_w = h, w
        
        # 如果没有完美的因式分解，尝试近似值
        if best_h * best_w != actual_tokens:
            print(f"281无法完美因式分解，寻找近似解...")
            # 281是质数，只能是1x281或281x1
            # 我们选择接近正方形的近似值
            sqrt_val = int(actual_tokens**0.5)  # 约16.76
            
            # 尝试17x17=289 (多8个), 16x17=272 (少9个), 17x16=272 (少9个)
            candidates = [
                (17, 17, 289),  # 多8个
                (16, 18, 288),  # 多7个  
                (17, 16, 272),  # 少9个
                (19, 15, 285),  # 多4个
            ]
            
            # 选择差异最小的
            best_candidate = min(candidates, key=lambda x: abs(x[2] - actual_tokens))
            best_h, best_w = best_candidate[0], best_candidate[1]
            
        H_visual, W_visual = best_h, best_w
        
        # 创建attention map
        grid_size = H_visual * W_visual
        if grid_size >= actual_tokens:
            # 网格大于实际tokens，需要填充
            attention_grid = np.zeros(grid_size)
            attention_grid[:actual_tokens] = attention_values[:actual_tokens]
            attention_map = attention_grid.reshape(H_visual, W_visual)
        else:
            # 网格小于实际tokens，需要截取
            attention_map = attention_values[:grid_size].reshape(H_visual, W_visual)
            print(f"警告: 截取了前{grid_size}个tokens")
    else:
        # 回退到理论尺寸
        H_visual, W_visual = H_theoretical, W_theoretical
        grid_size = H_visual * W_visual
        if len(attention_values) < grid_size:
            attention_grid = np.zeros(grid_size)
            attention_grid[:len(attention_values)] = attention_values
            attention_map = attention_grid.reshape(H_visual, W_visual)
        else:
            attention_map = attention_values[:grid_size].reshape(H_visual, W_visual)
    
    # # 归一化attention map
    # if attention_map.max() > attention_map.min():
    #     attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    # 创建token信息，只包含实际存在的tokens
    token_info = []
    num_actual_tokens = min(actual_tokens if actual_tokens else len(attention_values), len(attention_values))
    
    for i in range(num_actual_tokens):
        row = i // W_visual
        col = i % W_visual
        
        # 确保不越界
        if row < H_visual and col < W_visual:
            attention_value = attention_values[i]
            relative_row = row / H_visual
            relative_col = col / W_visual
            
            token_info.append({
                'token_idx': vision_start + i,
                'row': row,
                'col': col,
                'relative_pos': (relative_row, relative_col),
                'attention': attention_value,
                'is_actual_token': True
            })
    
    # 按注意力权重排序
    token_info.sort(key=lambda x: x['attention'], reverse=True)
    
    print(f"前5个最重要的tokens:")
    for i, info in enumerate(token_info):
        print(f"  {i+1}. Token {info['token_idx']}: 位置({info['row']}, {info['col']}), 权重={info['attention']:.6f}")
    
    return attention_map, token_info, H_visual, W_visual


def visualize_token_attention(image, token_info, H_visual, W_visual, save_path, top_k=40):

    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # 计算每个patch在原图中的位置
    patch_h = height / H_visual
    patch_w = width / W_visual
    
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))

    axes.imshow(img_array)
    actual_tokens = [info for info in token_info if info.get('is_actual_token', True)]
    display_count = min(top_k, len(actual_tokens))
    axes.set_title(f'Top-{display_count} Most Important Regions (Actual Tokens)')
    
    colors = plt.cm.viridis(np.linspace(1,0.2, display_count))

    for i, info in enumerate(actual_tokens[:display_count]):
        row, col = info['row'], info['col']
        attention = info['attention']
        
        # 计算patch在原图中的位置
        x1 = col * patch_w
        y1 = row * patch_h
        x2 = (col + 1) * patch_w
        y2 = (row + 1) * patch_h
        
        # 绘制高亮矩形
        rect = patches.Rectangle((x1, y1), patch_w, patch_h, 
                               linewidth=2, edgecolor=colors[i], 
                               facecolor=colors[i], alpha=0.8)
        axes.add_patch(rect)

    axes.axis('off')


    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    



def Qwen_fine_reasoner_Retrieval_scores_entropy(image_PIL: list, captions: list, is_saliency=False, I2T=False, T2I=False, 
                                               output_attention=False, save_attention_path=None):

    predict = []
    entropies_all = []  # 存储所有生成 token 的 entropy
    attention_maps = [] if output_attention else None
    True_id = tokenizer.convert_tokens_to_ids('True')
    
    for idx, img in enumerate(image_PIL):
        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": f"Does the image match the caption: <{captions}>? Please directly output True or False"},
                        #{"type": "text", "text": f"{captions}"},
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

            
            if output_attention:
                # 使用简单的前向传播获取注意力权重，而不是生成过程
                outputs = model(**inputs, output_attentions=True, return_dict=True)
                logits = outputs.logits
                attention_weights = outputs.attentions  # tuple of attention weights for each layer
                
                # 获取生成的token ID用于显示
                last_token_logits = logits[:, -1, :]
                predicted_token_id = torch.argmax(last_token_logits, dim=-1)
                generated_text = tokenizer.decode(predicted_token_id[0], skip_special_tokens=True)
                print(f"预测的下一个token: {generated_text}")
                
            else:
                logits = model(**inputs).logits
                attention_weights = None
                generated_text = ""
            
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
            seq = inputs['input_ids']
            
            if output_attention and attention_weights is not None:
                try:
                    
                    H_visual, W_visual = get_featuremap_size(img)
                    
                    layer_attentions = []

                    for layer_idx, layer_attention in enumerate(attention_weights):
                        layer_attention = layer_attention.float()

                        layer_attention_avg = layer_attention.mean(dim=1)  
                        
                        
                        layer_attentions.append(layer_attention_avg)
                    
                    all_layers_attention = torch.stack(layer_attentions, dim=0) 

                    averaged_attention = all_layers_attention.mean(dim=0) 
                    averaged_attention = torch.softmax(averaged_attention/0.2, dim=-1)  # 确保注意力权重是概率分布
                    seq_len = averaged_attention.shape[-1]
                    
                    seq = inputs['input_ids'][0].tolist()
                    vision_token_start = seq.index(151652) + 1  # 151652是image start token，实际vision tokens从下一个开始
                    vision_token_end = seq.index(151653)  # 151653是image end token
                    
                    # 对于前向传播，我们分析最后一个位置的token对视觉区域的注意力
                    # 这表示模型在当前位置时对图像各部分的关注
                    output_tokens_start = len(seq) - 1  # 使用最后一个token位置
                    print(f"分析位置 {output_tokens_start} 的token对图像的注意力")
                    
                    # 获取实际的vision token数量
                    actual_vision_tokens = vision_token_end - vision_token_start
                    print(f"实际vision token数量: {actual_vision_tokens}")
                    print(f"估算的H*W: {H_visual * W_visual}")

                    
                    if vision_token_end <= seq_len:
                        attention_map, token_info, H_actual, W_actual = get_vision_attention_map(
                            averaged_attention, vision_token_start, vision_token_end, H_visual, W_visual, 
                            actual_vision_tokens, output_tokens_start
                        )
                        attention_maps.append(attention_map)

                        if save_attention_path:
                            save_path_tokens = f"{save_attention_path}_img_{idx}_tokens.png"
                            
                            visualize_token_attention(img, token_info, H_actual, W_actual, save_path_tokens, top_k=actual_vision_tokens)
                            
                    else:
                        print(f"Warning: Vision token range ({vision_token_start}:{vision_token_end}) exceeds sequence length ({seq_len})")
                        attention_maps.append(None)
                except Exception as e:
                    print(f"Error processing attention for image {idx}: {e}")
                    attention_maps.append(None)

    try:
        max_score = max(predict)
        
        candidates = [i for i, score in enumerate(predict) if score == max_score]
        
        if len(candidates) == 1:
            best_index = candidates[0]
        else:
            candidate_entropies = [(i, entropies_all[i]) for i in candidates]
            best_index = min(candidate_entropies, key=lambda x: x[1])[0]

        if output_attention:
            return best_index, predict, entropies_all, attention_maps
        else:
            return best_index, predict, entropies_all
    except ValueError:
        if output_attention:
            return 0, None, None, None
        else:
            return 0, None, None


path = '/root/dws/MCS/Codes/Group13'
image_path = [os.path.join(path,f'topk_image_index_{i}.jpg') for i in range(5)]
image_PIL = []
for image in image_path:
    img = Image.open(image)
    image_PIL.append(img)
i =0 
captions = "A girl in karate uniform breaking a stick with a front kick."
import time
start_time = time.time()

# 启用注意力输出并保存可视化结果
predict_index, predict_scores, entropies, attention_maps = Qwen_fine_reasoner_Retrieval_scores_entropy(
    image_PIL, captions, 
    is_saliency=True, I2T=False, T2I=True, 
    output_attention=True, 
    save_attention_path="/root/dws/MCS/Codes/save_attentions/attention_visualization"
)

end_time = time.time()
print("Time taken:", end_time - start_time)
print("Predict index:", predict_index)
print("Predict scores:", predict_scores)
print("Entropies:", entropies)

# 显示注意力分析结果
if attention_maps:
    print(f"\n=== 注意力分析完成 ===")
    print(f"处理了 {len(attention_maps)} 张图像")
    for i, attention_map in enumerate(attention_maps):
        if attention_map is not None:
            print(f"图像 {i}: 注意力热力图已生成")
        else:
            print(f"图像 {i}: 注意力热力图生成失败")
else:
    print("未生成注意力热力图")