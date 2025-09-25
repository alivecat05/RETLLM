# CLIP 评估工具

本目录包含用于CLIP和Long-CLIP模型的多模态检索评估工具，专门适配M-BEIR数据集的真实格式。

## 文件说明

### 1. `mbeir_real_eval.py` - 真实M-BEIR数据格式评估器 ⭐ **推荐使用**
- 基于 `/root/dws/MCS/Datasets/M-BEIR` 的真实数据格式
- 支持所有M-BEIR任务的评估
- 包含完整的命令行接口和批量处理

### 2. `mbeir_clip_eval.py` - M-BEIR数据集专用评估器
- 修改后适配真实M-BEIR数据格式
- 支持从本地JSONL文件加载数据

### 3. `clip_eval.py` - 通用CLIP评估器
- 支持CLIP和Long-CLIP模型
- 提供完整的嵌入提取和检索评估流程

### 4. `simple_clip_eval.py` - 简化版评估工具
- 最易使用的评估脚本
- 支持批量处理和命令行参数

## M-BEIR 数据集结构

基于 `/root/dws/MCS/Datasets/M-BEIR` 的实际结构：

```
M-BEIR/
├── query/
│   ├── test/
│   │   ├── mbeir_mscoco_task0_test.jsonl
│   │   ├── mbeir_cirr_task7_test.jsonl
│   │   └── ...
│   └── val/
├── cand_pool/
│   ├── local/
│   │   ├── mbeir_mscoco_task0_test_cand_pool.jsonl
│   │   ├── mbeir_cirr_task7_cand_pool.jsonl
│   │   └── ...
│   └── global/
└── mbeir_images/
    ├── mscoco_images/
    ├── cirr_images/
    └── ...
```

### 数据格式

**查询文件格式** (`query/test/*.jsonl`):
```json
{
  "qid": "9:1",
  "query_txt": "A man with a red helmet on a small moped on a dirt road.",
  "query_img_path": null,
  "query_modality": "text",
  "pos_cand_list": ["9:1"],
  "neg_cand_list": [],
  "task_id": 0
}
```

**候选文件格式** (`cand_pool/local/*.jsonl`):
```json
{
  "txt": null,
  "img_path": "mbeir_images/mscoco_images/val2014/COCO_val2014_000000391895.jpg",
  "modality": "image",
  "did": "9:1"
}
```

## 快速开始

### 1. 使用真实M-BEIR评估器（推荐）

```bash
# 评估MS-COCO任务
python mbeir_real_eval.py --task mbeir_mscoco_task0 --clip_type CLIP

# 评估CIRR任务，使用Long-CLIP
python mbeir_real_eval.py --task mbeir_cirr_task7 --clip_type Long-CLIP --device cuda:1

# 强制重新计算嵌入
python mbeir_real_eval.py --task mbeir_webqa_task1 --force_recompute

# 查看所有可用任务
python mbeir_real_eval.py --help
```

### 2. 可用的M-BEIR任务

- `mbeir_mscoco_task0` - MS-COCO图像检索（文本→图像）
- `mbeir_mscoco_task3` - MS-COCO文本检索（图像→文本）
- `mbeir_cirr_task7` - CIRR组合图像检索
- `mbeir_fashioniq_task7` - FashionIQ时尚检索
- `mbeir_webqa_task1` - WebQA问答检索
- `mbeir_webqa_task2` - WebQA多模态检索
- `mbeir_visualnews_task0` - VisualNews新闻检索
- `mbeir_visualnews_task3` - VisualNews图像字幕
- `mbeir_infoseek_task6` - InfoSeek信息检索
- `mbeir_infoseek_task8` - InfoSeek知识检索
- `mbeir_fashion200k_task0` - Fashion200K时尚检索
- `mbeir_fashion200k_task3` - Fashion200K描述检索
- `mbeir_edis_task2` - EDIS文档检索
- `mbeir_nights_task4` - NIGHTS夜间图像检索
- `mbeir_oven_task6` - OVEN实体检索
- `mbeir_oven_task8` - OVEN知识图谱检索

### 3. 程序化使用

```python
from mbeir_real_eval import run_mbeir_evaluation

# 运行评估
results = run_mbeir_evaluation(
    mbeir_dir="/root/dws/MCS/Datasets/M-BEIR",
    task_name="mbeir_mscoco_task0",
    split="test",
    clip_type="CLIP",
    device="cuda:0",
    force_recompute=False
)

print(results)
# 输出: {'Recall@1': 0.4532, 'Recall@5': 0.7231, ...}
```

## 评估结果

评估完成后，会在 `/root/dws/MCS/Codes_MBEIR/results/{task_name}/` 目录下生成：

1. **`query_embeddings_{CLIP_TYPE}.pkl`** - 查询嵌入文件
2. **`candidate_embeddings_{CLIP_TYPE}.pkl`** - 候选嵌入文件  
3. **`results_{CLIP_TYPE}.json`** - 评估指标结果
4. **`detailed_results_{CLIP_TYPE}.json`** - 详细的每个查询结果
5. **`{task_name}_{CLIP_TYPE}_evaluation.log`** - 详细日志

### 结果格式示例

**评估指标** (`results_CLIP.json`):
```json
{
  "Recall@1": 0.4532,
  "Recall@5": 0.7231,
  "Recall@10": 0.8456,
  "Recall@20": 0.9123
}
```

**详细结果** (`detailed_results_CLIP.json`):
```json
[
  {
    "qid": "9:1",
    "query_txt": "A man with a red helmet...",
    "pos_cand_list": ["9:1"],
    "retrieved_did": "9:1",
    "is_correct": true
  },
  ...
]
```

## 配置参数

### 命令行参数

- `--mbeir_dir`: M-BEIR数据集目录（默认: `/root/dws/MCS/Datasets/M-BEIR`）
- `--task`: 要评估的任务名称（如: `mbeir_mscoco_task0`）
- `--split`: 数据分割（`test` 或 `val`，默认: `test`）
- `--clip_type`: CLIP模型类型（`CLIP` 或 `Long-CLIP`，默认: `CLIP`）
- `--device`: 计算设备（如: `cuda:0`，默认: `cuda:0`）
- `--force_recompute`: 强制重新计算嵌入（覆盖已有文件）

### 性能优化

- **批处理大小**: 默认32，可根据GPU内存调整
- **自动内存管理**: 自动清理GPU缓存
- **增量计算**: 自动跳过已计算的嵌入文件
- **混合精度**: 使用bfloat16加速计算

## 实际示例

### 评估MS-COCO图像检索任务

```bash
# 使用CLIP模型评估
python mbeir_real_eval.py 
    --task mbeir_mscoco_task0 
    --clip_type CLIP 
    --device cuda:0

# 预期输出:
# 初始化 CLIP 模型...
# 加载查询数据...
# 加载了 24810 个查询
# 加载候选数据...  
# 加载了 5001 个候选
# 提取查询嵌入...
# 提取候选嵌入...
# 计算检索指标...
# Recall@1: 0.4532 (11245/24810)
# Recall@5: 0.7231 (17943/24810)
# Recall@10: 0.8456 (20976/24810)
# Recall@20: 0.9123 (22632/24810)
```

### 评估CIRR组合图像检索任务

```bash
# 使用Long-CLIP模型评估（推荐用于复杂任务）
python mbeir_real_eval.py 
    --task mbeir_cirr_task7 
    --clip_type Long-CLIP 
    --device cuda:1
```

### 批量评估多个任务

```bash
# 创建批量评估脚本
cat > batch_eval.sh << 'EOF'
#!/bin/bash

tasks=("mbeir_mscoco_task0" "mbeir_cirr_task7" "mbeir_webqa_task1")
clip_types=("CLIP" "Long-CLIP")

for task in "${tasks[@]}"; do
    for clip_type in "${clip_types[@]}"; do
        echo "评估 $task 使用 $clip_type"
        python mbeir_real_eval.py 
            --task "$task" 
            --clip_type "$clip_type" 
            --device cuda:0
    done
done
EOF

chmod +x batch_eval.sh
./batch_eval.sh
```

## 故障排除

### 常见问题

1. **文件不存在错误**
   ```
   确保M-BEIR数据集已正确下载到指定目录
   检查任务名称是否正确拼写
   ```

2. **CUDA内存不足**
   ```bash
   # 减少批处理大小（修改代码中的batch_size参数）
   # 或使用CPU模式
   python mbeir_real_eval.py --device cpu
   ```

3. **模型加载失败**
   ```
   检查CLIP/Long-CLIP模型路径是否正确
   确保相关依赖已安装
   ```

## 性能基准

基于RTX 4090的性能参考：

| 任务 | 查询数量 | 候选数量 | CLIP时间 | Long-CLIP时间 |
|------|----------|----------|----------|---------------|
| mbeir_mscoco_task0 | 24,810 | 5,001 | ~15分钟 | ~25分钟 |
| mbeir_cirr_task7 | 4,171 | 25,000 | ~8分钟 | ~15分钟 |
| mbeir_webqa_task1 | 2,878 | 4,000 | ~5分钟 | ~10分钟 |

## 许可证

本工具遵循MIT许可证。CLIP和Long-CLIP模型的多模态检索评估工具。

## 文件说明

### 1. `clip_eval.py` - 通用CLIP评估器
- 支持CLIP和Long-CLIP模型
- 提供完整的嵌入提取和检索评估流程
- 支持图像、文本和多模态特征提取

### 2. `mbeir_clip_eval.py` - M-BEIR数据集专用评估器  
- 专门针对M-BEIR数据集格式设计
- 支持从HuggingFace加载M-BEIR数据集
- 自动处理候选池提取和查询匹配

### 3. `simple_clip_eval.py` - 简化版评估工具
- 最易使用的评估脚本
- 支持批量处理和命令行参数
- 提供灵活的配置选项

## 安装依赖

```bash
pip install torch torchvision datasets transformers pillow tqdm scikit-learn
```

## 使用方法

### 基本用法 - 使用简化版评估工具

```python
from simple_clip_eval import run_evaluation

# 配置评估参数
config = {
    'clip_type': 'CLIP',  # 或 'Long-CLIP'
    'device': 'cuda:0',
    'img_dir': '/path/to/images',
    'output_dir': '/root/dws/MCS/Codes_MBEIR/results/my_eval',
    'batch_size': 32,
    'encoding_mode': 'multimodal',  # 'multimodal', 'text', 'image'
    'k_values': [1, 5, 10],
    'force_recompute': False,
    
    # 数据格式
    'query_data': [
        {'text': '查询文本1', 'image': 'query1.jpg'},
        {'text': '查询文本2', 'image': 'query2.jpg'},
    ],
    'candidate_data': [
        {'text': '候选文本1', 'image': 'cand1.jpg'},
        {'text': '候选文本2', 'image': 'cand2.jpg'},
        {'text': '候选文本3', 'image': 'cand3.jpg'},
    ],
    'ground_truth': [0, 1]  # 正确答案索引
}

# 运行评估
results = run_evaluation(config)
print(results)
```

### 命令行用法

```bash
# 使用CLIP模型评估
python simple_clip_eval.py --clip_type CLIP --device cuda:0

# 使用Long-CLIP模型评估
python simple_clip_eval.py --clip_type Long-CLIP --device cuda:1 --force_recompute

# 指定输出目录
python simple_clip_eval.py --output_dir /path/to/results
```

### M-BEIR数据集评估

```python
# 使用M-BEIR专用评估器
python mbeir_clip_eval.py

# 在脚本中修改以下参数：
# - dataset_name: M-BEIR数据集名称
# - subset_name: 子数据集名称（如 "mscoco", "fashioniq"）
# - img_dir: 图像文件目录
# - clip_type: "CLIP" 或 "Long-CLIP"
```

## 数据格式

### 查询数据格式
```python
query_data = [
    {
        'text': '查询文本（可选）',
        'image': '图像路径（可选）'
    },
    # 更多查询...
]
```

### 候选数据格式
```python
candidate_data = [
    {
        'text': '候选文本（可选）',
        'image': '图像路径（可选）'
    },
    # 更多候选...
]
```

### 标准答案格式
```python
ground_truth = [0, 2, 1]  # 每个查询对应的正确候选索引
```

## 输出结果

评估完成后，会在指定输出目录生成以下文件：

1. `query_embeddings_[CLIP_TYPE].pkl` - 查询嵌入文件
2. `candidate_embeddings_[CLIP_TYPE].pkl` - 候选嵌入文件  
3. `results_[CLIP_TYPE].json` - 评估结果文件
4. `evaluation_[CLIP_TYPE].log` - 详细日志文件

### 结果格式示例
```json
{
  "Recall@1": 0.7500,
  "Recall@5": 0.9000,
  "Recall@10": 0.9500
}
```

## 高级配置

### 模型选择
- `clip_type`: 选择 "CLIP" 或 "Long-CLIP"
- `device`: 指定GPU设备，如 "cuda:0"

### 编码模式
- `multimodal`: 图像+文本融合（推荐）
- `text`: 仅使用文本特征
- `image`: 仅使用图像特征

### 批处理
- `batch_size`: 批处理大小，根据GPU内存调整
- 默认32，大模型或小显存建议设置为16或8

### 评估指标
- `k_values`: 计算Recall@K的K值列表
- 默认 [1, 5, 10]，可根据需要调整

## 注意事项

1. **模型路径**: 确保CLIP模型路径正确
   - CLIP: 自动下载 "ViT-L/14@336px"
   - Long-CLIP: 需要本地模型文件 "/root/dws/MCS/Long-CLIP/checkpoints/longclip-L.pt"

2. **图像路径**: 确保图像文件存在且可访问
   - 支持相对路径和绝对路径
   - 自动处理图像加载错误

3. **内存管理**: 大数据集评估时注意内存使用
   - 启用了自动内存清理
   - 可调整batch_size减少内存占用

4. **GPU使用**: 确保指定的GPU设备可用
   - 自动检测CUDA可用性
   - 支持CPU模式（较慢）

## 故障排除

### 常见问题

1. **模型加载失败**
   ```
   检查模型路径是否正确
   确保相关依赖已安装
   ```

2. **图像加载错误**
   ```
   检查图像文件是否存在
   确保图像格式受支持（jpg, png等）
   ```

3. **内存不足**
   ```
   减小batch_size
   使用较小的模型
   清理无用变量
   ```

4. **CUDA错误**
   ```
   检查GPU可用性
   调整device参数
   更新CUDA驱动
   ```

## 扩展开发

### 添加新的融合方法
在 `SimpleCLIPEvaluator.encode_multimodal()` 中添加新的 `fusion_method`：

```python
elif fusion_method == "custom_fusion":
    # 自定义融合逻辑
    combined = custom_fusion_function(text_features, image_features)
```

### 添加新的评估指标
在 `compute_retrieval_metrics()` 中添加新指标：

```python
# 计算MRR (Mean Reciprocal Rank)
mrr_scores = []
for i, gt_idx in enumerate(ground_truth_indices):
    rank = torch.where(top_k_indices[i] == gt_idx)[0]
    if len(rank) > 0:
        mrr_scores.append(1.0 / (rank[0].item() + 1))
    else:
        mrr_scores.append(0.0)
results['MRR'] = np.mean(mrr_scores)
```

## 许可证

本工具遵循MIT许可证。
