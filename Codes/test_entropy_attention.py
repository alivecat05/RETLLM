#!/usr/bin/env python3
"""
简单测试Qwen_fine_reasoner_Retrieval_scores_entropy注意力可视化功能
"""

import sys
import os
from PIL import Image
import torch

# 添加路径
sys.path.append('/root/dws/MCS/Codes')

try:
    from Qwen_vl import Qwen_fine_reasoner_Retrieval_scores_entropy
    print("✓ 成功导入 Qwen_fine_reasoner_Retrieval_scores_entropy")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 创建一个简单的测试图像
    test_image = Image.new('RGB', (224, 224), color='lightblue')
    print("✓ 创建测试图像")
    
    # 测试基本调用（不输出注意力）
    try:
        result = Qwen_fine_reasoner_Retrieval_scores_entropy(
            image_PIL=[test_image],
            captions="A blue image",
            output_attention=False
        )
        print(f"✓ 基本调用成功，返回长度: {len(result)}")
        print(f"  结果: {result}")
    except Exception as e:
        print(f"✗ 基本调用失败: {e}")
        return False
    
    return True

def test_attention_functionality():
    """测试注意力功能"""
    print("\n=== 测试注意力功能 ===")
    
    # 创建测试图像
    test_image = Image.new('RGB', (224, 224), color='lightgreen')
    
    try:
        result = Qwen_fine_reasoner_Retrieval_scores_entropy(
            image_PIL=[test_image],
            captions="A green test image",
            output_attention=True,
            save_attention_path="/tmp/test_attention"
        )
        
        if len(result) == 4:
            best_index, predict, entropies, attention_maps = result
            print(f"✓ 注意力调用成功")
            print(f"  最佳索引: {best_index}")
            print(f"  预测分数: {predict}")
            print(f"  熵值: {entropies}")
            print(f"  注意力图数量: {len(attention_maps) if attention_maps else 0}")
            
            if attention_maps:
                for i, att_map in enumerate(attention_maps):
                    if att_map is not None:
                        print(f"  注意力图 {i}: {att_map.shape}")
                    else:
                        print(f"  注意力图 {i}: None")
        else:
            print(f"✗ 返回结果长度错误: {len(result)}")
            return False
            
    except Exception as e:
        print(f"✗ 注意力调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """主函数"""
    print("Qwen_fine_reasoner_Retrieval_scores_entropy 注意力功能测试")
    print("=" * 60)
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.device_count()} 个设备")
    else:
        print("⚠ CUDA不可用，使用CPU")
    
    # 测试基本功能
    if not test_basic_functionality():
        print("\n✗ 基本功能测试失败")
        return
    
    # 测试注意力功能
    if not test_attention_functionality():
        print("\n✗ 注意力功能测试失败")
        return
    
    print("\n✓ 所有测试通过！")
    print("\n使用示例:")
    print("""
# 基本用法
result = Qwen_fine_reasoner_Retrieval_scores_entropy(
    image_PIL=[your_image],
    captions="your caption"
)

# 带注意力可视化
result = Qwen_fine_reasoner_Retrieval_scores_entropy(
    image_PIL=[your_image],
    captions="your caption",
    output_attention=True,
    save_attention_path="/path/to/save/attention"
)

# 解析结果
if len(result) == 4:  # 包含注意力
    best_index, scores, entropies, attention_maps = result
else:  # 不包含注意力
    best_index, scores, entropies = result
""")

if __name__ == "__main__":
    main()
