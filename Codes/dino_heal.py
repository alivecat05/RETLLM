from modelscope import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
import torch.nn.functional as F

path = '/root/dws/MCS/Models/dinov2-large'

processor = AutoImageProcessor.from_pretrained(path)
model = AutoModel.from_pretrained(path)

print('Dino model loaded')
min_pixels = 256 * 28 * 28
max_pixels=512*28*28
def get_saliency_map(image, H_visual, W_visual): 
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs,output_attentions=True)
    attention = outputs.attentions[-1]
    mean_attention = attention.mean(dim=1)
    
    class_attention = mean_attention[:, 0, 1:]

    num_patches = class_attention.shape[-1]
    spatial_size = int(num_patches ** 0.5)
    attn_map = class_attention.view(1, 1, spatial_size, spatial_size)  # [1, 1, H', W']


    upsampled_map = F.interpolate(attn_map, size=(H_visual, W_visual), mode='bilinear', align_corners=False)
    upsampled_map = torch.flatten(upsampled_map, start_dim=2)  # [1, 1, H*W]
    saliency_map = torch.sigmoid(upsampled_map)  # 激活归一化到 0~1

    return saliency_map  # shape: [1, 1, H, W]，
def get_featuremap_size(img):
    from math import sqrt
    import math
    patch_size = 14
    merge_size =2 
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
# H_visual, W_visual = get_featuremap_size(image)
# get_saliency_map(image, H_visual, W_visual)