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
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
model_id = '/root/dws/MCS/Models/Qwen2.5-VL-7B-Instruct'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
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
    """
    获取图像经过模型处理后的特征图尺寸
    注意：Qwen2.5-VL会进行token合并，所以实际token数量可能小于理论值
    """
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
    """
    从注意力权重中提取视觉token的注意力图
    获取指定位置token对输入图像的注意力分布
    """
    attention_weights = attention_weights.float()
    
    # 平均所有注意力头
    attention_weights = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
    
    # 使用指定位置的token对视觉tokens的注意力
    if output_tokens_start is not None:
        # 使用指定位置的token
        aggregated_attention = attention_weights[0, output_tokens_start, vision_start:vision_end]  # [num_vision_tokens]
        print(f"使用位置 {output_tokens_start} 的token对视觉区域的注意力")
    else:
        # 默认使用最后一个token对视觉tokens的注意力
        aggregated_attention = attention_weights[0, -1, vision_start:vision_end]  # [num_vision_tokens]
        print("使用最后一个token的注意力")

    attention_values = aggregated_attention.detach().cpu().numpy()
    
    # 检查是否有NaN值
    if np.isnan(attention_values).any():
        print("警告: 发现NaN值，将其替换为0")
        attention_values = np.nan_to_num(attention_values, nan=0.0)
    
    print(f"实际获取的attention长度: {len(attention_values)}")
    print(f"理论token数量 (H*W): {H_theoretical * W_theoretical}")
    print(f"实际token数量: {actual_tokens}")
    
    # 分析注意力分布
    print(f"注意力权重统计:")
    print(f"  最大值: {attention_values.max():.6f}")
    print(f"  最小值: {attention_values.min():.6f}")
    print(f"  平均值: {attention_values.mean():.6f}")
    print(f"  标准差: {attention_values.std():.6f}")
    print(f"  是否包含NaN: {np.isnan(attention_values).any()}")
    print(f"  是否包含Inf: {np.isinf(attention_values).any()}")
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
    
    # 归一化attention map
    if attention_map.max() > attention_map.min():
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
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
    for i, info in enumerate(token_info[:5]):
        print(f"  {i+1}. Token {info['token_idx']}: 位置({info['row']}, {info['col']}), 权重={info['attention']:.6f}")
    
    return attention_map, token_info, H_visual, W_visual


def visualize_attention_on_image(image, attention_map, save_path):
    """改进的注意力可视化函数"""
    img_array = np.array(image)
    
    # 确保attention_map是float32类型
    attention_map = attention_map.astype(np.float32)
    
    # 使用双线性插值将注意力图调整到与图像相同的尺寸
    from scipy.ndimage import zoom
    h_ratio = img_array.shape[0] / attention_map.shape[0]
    w_ratio = img_array.shape[1] / attention_map.shape[1]
    attention_resized = zoom(attention_map, (h_ratio, w_ratio), order=1)
    
    # 确保尺寸完全匹配
    if attention_resized.shape != img_array.shape[:2]:
        attention_resized = cv2.resize(attention_resized, (img_array.shape[1], img_array.shape[0]))
    
    # 应用高斯平滑以获得更自然的热力图
    from scipy.ndimage import gaussian_filter
    attention_smooth = gaussian_filter(attention_resized, sigma=8)
    
    # 增强对比度
    attention_enhanced = np.power(attention_smooth, 0.7)  # 伽马校正
    
    # 创建彩色热力图
    heatmap = plt.cm.jet(attention_enhanced)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 叠加图像和热力图
    alpha = 0.5  # 增加透明度以更好地看到原图
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap, alpha, 0)
    
    # 保存结果
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_array)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(attention_map, cmap='jet', interpolation='bilinear')
    plt.title(f'Output-to-Vision Attention\n({attention_map.shape[0]}x{attention_map.shape[1]})')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(attention_enhanced, cmap='jet')
    plt.title('Enhanced Attention\n(Output→Image)')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title('Model Focus During Response')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"注意力叠加图已保存到: {save_path}")


def save_attention_heatmap(attention_map, save_path):

    attention_map = attention_map.astype(np.float32)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_map, cmap='jet', interpolation='bilinear')
    plt.colorbar(label='Attention Weight')
    plt.title('Attention Heatmap')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"注意力热力图已保存到: {save_path}")


def visualize_token_attention(image, token_info, H_visual, W_visual, save_path, top_k=10):

    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # 计算每个patch在原图中的位置
    patch_h = height / H_visual
    patch_w = width / W_visual
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 原图
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. 网格划分
    axes[0, 1].imshow(img_array)
    axes[0, 1].set_title(f'Grid Division ({H_visual}x{W_visual})')
    
    # 绘制网格线
    for i in range(H_visual + 1):
        y = i * patch_h
        axes[0, 1].axhline(y=y, color='white', linewidth=1, alpha=0.7)
    for j in range(W_visual + 1):
        x = j * patch_w
        axes[0, 1].axvline(x=x, color='white', linewidth=1, alpha=0.7)
    
    axes[0, 1].axis('off')
    
    # 3. Top-K重要区域高亮（只显示实际存在的tokens）
    axes[1, 0].imshow(img_array)
    actual_tokens = [info for info in token_info if info.get('is_actual_token', True)]
    display_count = min(top_k, len(actual_tokens))
    axes[1, 0].set_title(f'Top-{display_count} Most Important Regions (Actual Tokens)')
    
    # 创建颜色映射
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, display_count))
    
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
                               facecolor=colors[i], alpha=0.6)
        axes[1, 0].add_patch(rect)
        
        # 添加排名标签
        axes[1, 0].text(x1 + patch_w/2, y1 + patch_h/2, f'{i+1}', 
                       ha='center', va='center', fontsize=12, 
                       fontweight='bold', color='white')
    
    axes[1, 0].axis('off')
    
    # 4. 注意力权重分布
    attentions = [info['attention'] for info in actual_tokens]
    x_positions = range(len(attentions))
    
    bars = axes[1, 1].bar(x_positions, attentions, color='skyblue', alpha=0.7)
    axes[1, 1].set_xlabel('Token Index (Actual Tokens Only)')
    axes[1, 1].set_ylabel('Attention Weight')
    axes[1, 1].set_title(f'Attention Weight Distribution ({len(attentions)} tokens)')
    
    # 高亮前top_k
    for i in range(min(display_count, len(attentions))):
        bars[i].set_color('red')
        bars[i].set_alpha(0.8)
    
    # 设置x轴标签，显示每隔几个
    if len(attentions) > 20:
        step = len(attentions) // 10
        axes[1, 1].set_xticks(x_positions[::step])
        axes[1, 1].set_xticklabels([str(i) for i in x_positions[::step]])
    
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
            # 处理注意力可视化
            if output_attention and attention_weights is not None:
                try:
                    # 获取图像的特征图尺寸
                    H_visual, W_visual = get_featuremap_size(img)
                    
                    # 提取最后一层的注意力权重
                    last_layer_attention = attention_weights[-1]  # [batch_size, num_heads, seq_len, seq_len]
                    
                    # 获取实际的视觉token位置
                    seq_len = last_layer_attention.shape[-1]
                    
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
                            last_layer_attention, vision_token_start, vision_token_end, H_visual, W_visual, 
                            actual_vision_tokens, output_tokens_start
                        )
                        attention_maps.append(attention_map)
                        
                        
                        # 可视化并保存
                        if save_attention_path:
                            save_path_img = f"{save_attention_path}_img_{idx}_overlay.png"
                            save_path_heatmap = f"{save_attention_path}_img_{idx}_heatmap.png"
                            save_path_tokens = f"{save_attention_path}_img_{idx}_tokens.png"
                            
                            # 保存叠加图（添加生成的文本信息）
                            visualize_attention_on_image(img, attention_map, save_path_img)
                            
                            # 保存热力图
                            save_attention_heatmap(attention_map, save_path_heatmap)
                            
                            # 保存token位置图
                            visualize_token_attention(img, token_info, H_actual, W_actual, save_path_tokens)
                            
                            print(f"=== 图像 {idx} 注意力分析完成 ===")
                            print(f"模型回答: {generated_text}")
                            print(f"注意力热力图显示了模型在回答时关注的图像区域")
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


path = '/root/dws/MCS/Codes/Group0'
image_path = [os.path.join(path,f'topk_image_index_{i}.jpg') for i in range(5)]
image_PIL = []
for image in image_path:
    img = Image.open(image)
    image_PIL.append(img)
i =0 
captions = "The man with pierced ears is wearing glasses and an orange hat."
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