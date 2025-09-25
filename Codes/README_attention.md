# Qwen VL 注意力可视化功能

这个文档介绍如何使用修改后的注意力可视化函数来输出和可视化模型的注意力权重。

## 支持的函数

目前支持注意力可视化的函数包括：
1. `Qwen_fine_reasoner_Retrieval_scores_entropy_saliency` - 复杂的图像-文本匹配函数
2. `Qwen_fine_reasoner_Retrieval_scores_entropy` - 简单的图像-文本匹配函数

## 功能概述

新增的注意力可视化功能可以帮助你：
1. **提取注意力权重**：获取模型在处理图像时的注意力分布
2. **可视化注意力图**：将注意力权重叠加到原图像上显示
3. **分析模型关注点**：了解模型在做决策时关注图像的哪些区域

## 安装依赖

确保安装了以下依赖库：

```bash
pip install matplotlib opencv-python numpy torch torchvision PIL
```

## 使用方法

### 1. 基本用法 - Qwen_fine_reasoner_Retrieval_scores_entropy

```python
from Qwen_vl import Qwen_fine_reasoner_Retrieval_scores_entropy
from PIL import Image

# 加载图像
image_path = "/path/to/your/image.jpg"
image = Image.open(image_path).convert('RGB')
image_PIL = [image]  # 简单的图像列表

# 设置查询
captions = "your query text"  # 直接的字符串

# 调用函数并启用注意力可视化
result = Qwen_fine_reasoner_Retrieval_scores_entropy(
    image_PIL=image_PIL,
    captions=captions,
    output_attention=True,  # 启用注意力输出
    save_attention_path="/path/to/save/attention"  # 可选：自动保存路径
)

# 解析结果
if len(result) == 4:  # 包含注意力信息
    best_index, predict_scores, entropies, attention_maps = result
    print(f"最佳匹配索引: {best_index}")
    print(f"预测分数: {predict_scores}")
    print(f"熵值: {entropies}")
    print(f"注意力图数量: {len(attention_maps) if attention_maps else 0}")
else:
    best_index, predict_scores, entropies = result
    print("未启用注意力输出")
```

### 2. 复杂用法 - Qwen_fine_reasoner_Retrieval_scores_entropy_saliency

```python
from Qwen_vl import Qwen_fine_reasoner_Retrieval_scores_entropy_saliency
from PIL import Image

# 加载图像
image_path = "/path/to/your/image.jpg"
image = Image.open(image_path).convert('RGB')
image_PIL = [(image, "optional_image_caption")]

# 设置查询
captions = ["your query text", None]  # [query_text, query_image_or_None]

# 调用函数并启用注意力可视化
result = Qwen_fine_reasoner_Retrieval_scores_entropy_saliency(
    image_PIL=image_PIL,
    captions=captions,
    output_attention=True,  # 启用注意力输出
    save_attention_path="/path/to/save/attention"  # 可选：自动保存路径
)

# 解析结果
if len(result) == 4:  # 包含注意力信息
    best_index, predict_scores, entropies, attention_maps = result
    print(f"最佳匹配索引: {best_index}")
    print(f"预测分数: {predict_scores}")
    print(f"熵值: {entropies}")
    print(f"注意力图数量: {len(attention_maps) if attention_maps else 0}")
else:
    best_index, predict_scores, entropies = result
    print("未启用注意力输出")
```

### 2. 手动可视化特定图像

```python
from Qwen_vl import visualize_attention_on_image, save_attention_heatmap

# 假设你已经获得了attention_maps
if attention_maps and attention_maps[0] is not None:
    # 可视化注意力叠加图
    visualize_attention_on_image(
        image,  # 原始PIL图像
        attention_maps[0],  # 对应的注意力图
        "/path/to/save/custom_attention.png",  # 保存路径
        alpha=0.6  # 透明度
    )
    
    # 保存纯热力图
    save_attention_heatmap(
        attention_maps[0],
        "/path/to/save/heatmap.png"
    )
```

### 3. 批量处理多个图像

```python
# 对于 Qwen_fine_reasoner_Retrieval_scores_entropy
images = [Image.open(f"image{i}.jpg") for i in range(3)]

result = Qwen_fine_reasoner_Retrieval_scores_entropy(
    image_PIL=images,
    captions="query text",
    output_attention=True,
    save_attention_path="/output/batch_attention"
)

# 对于 Qwen_fine_reasoner_Retrieval_scores_entropy_saliency
images = [
    (Image.open("image1.jpg"), "caption1"),
    (Image.open("image2.jpg"), "caption2"),
    (Image.open("image3.jpg"), "caption3")
]

result = Qwen_fine_reasoner_Retrieval_scores_entropy_saliency(
    image_PIL=images,
    captions=["query text", None],
    output_attention=True,
    save_attention_path="/output/batch_attention"
)

if len(result) == 4:
    best_index, scores, entropies, attention_maps = result
    
    # 分析每个图像的注意力
    for i, att_map in enumerate(attention_maps):
        if att_map is not None:
            print(f"图像 {i} 注意力范围: [{att_map.min():.4f}, {att_map.max():.4f}]")
            # 进一步分析...
```

## 函数参数说明

### Qwen_fine_reasoner_Retrieval_scores_entropy

**基础参数：**
- `image_PIL` (list): PIL图像列表
- `captions` (str): 查询文本字符串
- `is_saliency` (bool): 是否使用显著性，默认 False
- `I2T` (bool): 图像到文本模式，默认 False
- `T2I` (bool): 文本到图像模式，默认 False

**新增参数：**
- `output_attention` (bool): 是否输出注意力权重，默认 False
- `save_attention_path` (str): 自动保存注意力可视化的路径前缀，可选

**返回值：**
- 当 `output_attention=False` 时：`(best_index, predict_scores, entropies)`
- 当 `output_attention=True` 时：`(best_index, predict_scores, entropies, attention_maps)`

### Qwen_fine_reasoner_Retrieval_scores_entropy_saliency

**基础参数：**
- `image_PIL` (list): 图像元组列表 [(image, caption), ...]
- `captions` (list): [查询文本, 查询图像或None]
- `is_saliency` (bool): 是否使用显著性，默认 False
- `I2T` (bool): 图像到文本模式，默认 False
- `T2I` (bool): 文本到图像模式，默认 False

**新增参数：**
- `output_attention` (bool): 是否输出注意力权重，默认 False
- `save_attention_path` (str): 自动保存注意力可视化的路径前缀，可选

**返回值：**
- 当 `output_attention=False` 时：`(best_index, predict_scores, entropies)`
- 当 `output_attention=True` 时：`(best_index, predict_scores, entropies, attention_maps)`

### 可视化函数

#### visualize_attention_on_image
```python
visualize_attention_on_image(image, attention_map, save_path=None, alpha=0.6)
```
- `image`: PIL图像对象
- `attention_map`: 注意力权重张量 [H, W]
- `save_path`: 保存路径（可选）
- `alpha`: 热力图透明度 (0-1)

#### save_attention_heatmap
```python
save_attention_heatmap(attention_map, save_path)
```
- `attention_map`: 注意力权重张量 [H, W]
- `save_path`: 保存路径

## 输出文件说明

当设置 `save_attention_path` 时，会自动生成：
- `{save_attention_path}_img_{idx}_overlay.png`: 注意力叠加图
- `{save_attention_path}_img_{idx}_heatmap.png`: 纯注意力热力图

## 运行演示

```bash
cd /root/dws/MCS/Codes
python attention_demo.py
```

这个演示脚本会：
1. 创建测试图像
2. 运行注意力可视化
3. 保存结果到 `/tmp/` 目录
4. 展示如何手动创建可视化

## 注意事项

1. **GPU内存**：启用注意力输出会消耗额外的GPU内存
2. **视觉token位置**：当前实现假设视觉tokens在序列开始位置，可能需要根据具体模型调整
3. **注意力层选择**：当前使用最后一层的注意力，你可以修改代码使用其他层
4. **分辨率**：注意力图的分辨率取决于模型的视觉特征图大小

## 调试和优化

如果注意力可视化效果不理想，可以：

1. **调整视觉token位置**：
```python
# 在get_vision_attention_map函数中调整
vision_token_start = 1  # 根据实际情况调整
vision_token_end = vision_token_start + num_vision_tokens
```

2. **使用不同的注意力层**：
```python
# 使用倒数第二层的注意力
second_last_attention = attention_weights[-2]
```

3. **调整注意力头的聚合方式**：
```python
# 不同的聚合方式
avg_attention = torch.mean(last_token_attention, dim=0)  # 平均
max_attention = torch.max(last_token_attention, dim=0)[0]  # 最大值
# 或选择特定的注意力头
specific_head_attention = last_token_attention[head_idx]
```

## 常见问题

**Q: 为什么注意力图是空的或全零？**
A: 可能是视觉token位置计算错误，请检查并调整 `vision_token_start` 和 `vision_token_end`。

**Q: 注意力图看起来很均匀，没有明显的关注点？**
A: 尝试使用不同的注意力层或调整可视化参数。

**Q: 内存不足错误？**
A: 减少批处理大小或在更强的GPU上运行。

## 示例结果

正确运行后，你会得到：
- 原始图像和注意力叠加的对比图
- 显示模型关注区域的热力图
- 注意力权重的数值统计信息

这些可视化结果能帮助你理解模型是如何"看"图像的，以及它在做决策时关注了哪些重要区域。
