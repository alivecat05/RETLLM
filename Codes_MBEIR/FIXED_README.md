# 修复后的多卡推理使用指南

## 问题解决

已修复的主要问题：
1. **DistributedDataParallel错误**: 添加了`get_model()`方法来正确访问被包装的模型
2. **accelerate库缺失**: 添加了回退机制，没有accelerate也能运行
3. **模型方法访问**: 使用`.module`属性访问原始模型的方法

## 使用方法

### 方法1: 安装accelerate库（推荐）

```bash
# 1. 安装accelerate
./install_accelerate.sh

# 2. 运行多卡推理
./run_multicard.sh
```

### 方法2: 使用单GPU（如果多卡有问题）

```bash
# 直接运行单GPU版本
./run_single.sh
```

### 方法3: 手动安装accelerate

```bash
pip3 install accelerate
python3 -c "import accelerate; print('安装成功')"
./run_multicard.sh
```

## 代码修改说明

### 1. 添加了安全的模型访问方法

```python
def get_model(self, model):
    """安全地获取原始模型，处理DistributedDataParallel包装"""
    return model.module if hasattr(model, 'module') else model
```

### 2. 修复了特征提取中的模型调用

```python
# 获取原始模型
clip_model = self.get_model(self.clip_model)
searle_model = self.get_model(self.searle_model)

# 现在可以正常调用方法
image_features = clip_model.encode_image(image_input)
```

### 3. 添加了回退机制

如果没有accelerate库，会自动使用单GPU模式，确保代码能正常运行。

## 验证安装

运行以下命令验证accelerate是否正确安装：

```bash
python3 -c "from accelerate import Accelerator; print('✅ accelerate可用')"
```

## 故障排除

1. **如果仍然有DistributedDataParallel错误**:
   - 检查是否正确使用了`get_model()`方法
   - 尝试单GPU模式：`./run_single.sh`

2. **如果accelerate命令未找到**:
   - 运行：`./install_accelerate.sh`
   - 或手动安装：`pip3 install accelerate`

3. **如果GPU内存不足**:
   - 修改batch_size参数（在代码中搜索batch_size并减小值）
   - 使用更少的GPU：修改`CUDA_VISIBLE_DEVICES`

现在代码应该能正常运行了！
