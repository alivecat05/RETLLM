#!/usr/bin/env python3
"""
简单的rerank正确率计算脚本
"""

import sys
import json
from mbeir_clip_eval import calculate_rerank_accuracy, load_topk_for_rerank

def example_rerank_accuracy():
    """
    展示如何计算rerank后的正确率
    """
    print("=== Rerank正确率计算示例 ===\n")
    
    # 示例1: 加载现有的top-k结果
    topk_file = "/root/dws/MCS/Codes_MBEIR/embeddings/mbeir_infoseek_task6/topk_for_rerank_Long-CLIP.json"
    
    try:
        # 加载原始top-k结果
        print(f"正在加载: {topk_file}")
        topk_results = load_topk_for_rerank(topk_file)
        print(f"已加载 {len(topk_results)} 个查询的结果\n")
        
        # 创建模拟的rerank结果（这里您需要替换为实际的rerank结果）
        print("创建模拟rerank结果（实际使用时请替换为真实的rerank结果）...")
        reranked_results = simulate_rerank_results(topk_results)
        
        # 计算rerank后的正确率
        print("\n正在计算rerank后的正确率...")
        accuracy_results = calculate_rerank_accuracy(reranked_results, k_values=[1, 5, 10])
        
        # 打印结果
        print("\n🎯 Rerank正确率结果:")
        print("-" * 40)
        for metric, value in accuracy_results.items():
            print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
        
        # 保存结果
        save_path = "/root/dws/MCS/Codes_MBEIR/rerank_accuracy_results.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(accuracy_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {save_path}")
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {topk_file}")
        print("请先运行 mbeir_clip_eval.py 生成top-k结果")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

def simulate_rerank_results(topk_results):
    """
    模拟rerank结果（仅用于演示）
    实际使用时，这里应该是您的Qwen VL rerank函数的输出
    """
    reranked_results = []
    
    for query_result in topk_results:
        # 复制原始查询信息
        reranked_query = {
            'qid': query_result['qid'],
            'query_txt': query_result['query_txt'],
            'pos_cand_list': query_result['pos_cand_list'],
            'reranked_candidates': []
        }
        
        # 模拟rerank：随机打乱候选顺序（实际使用时请替换为真实rerank）
        import random
        candidates = query_result['top_k_candidates'].copy()
        random.shuffle(candidates)
        
        # 转换为rerank格式
        for rank, candidate in enumerate(candidates):
            reranked_candidate = {
                'candidate_id': candidate['did'],  # 注意这里使用candidate_id
                'rank': rank + 1,
                'rerank_score': random.random()  # 模拟rerank分数
            }
            reranked_query['reranked_candidates'].append(reranked_candidate)
        
        reranked_results.append(reranked_query)
    
    return reranked_results

def real_usage_example():
    """
    真实使用场景的示例代码
    """
    print("\n=== 真实使用场景示例 ===")
    print("""
# 1. 加载top-k结果
topk_results = load_topk_for_rerank("path/to/topk_for_rerank_Long-CLIP.json")

# 2. 使用Qwen VL进行rerank（您需要实现这部分）
# reranked_results = your_qwen_rerank_function(topk_results)

# 3. 计算正确率
accuracy_results = calculate_rerank_accuracy(reranked_results, k_values=[1, 5, 10])

# 4. 查看结果
for metric, value in accuracy_results.items():
    print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
    """)

if __name__ == "__main__":
    example_rerank_accuracy()
    real_usage_example()
