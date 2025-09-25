#!/usr/bin/env python3
# 测试数据加载脚本

import sys
import os
sys.path.insert(0, '/root/dws/MCS')

import json

def test_data_loading():
    """测试M-BEIR数据加载是否正常"""
    
    # 测试数据路径
    mbeir_dir = "/root/dws/MCS/Datasets/M-BEIR"
    task_name = "mbeir_webqa_task2"
    split = "test"
    
    # 构造文件路径
    query_file = os.path.join(mbeir_dir, "query", split, f"{task_name}_{split}.jsonl")
    cand_file = os.path.join(mbeir_dir, "cand_pool", "local", f"{task_name}_cand_pool.jsonl")
    
    print(f"测试文件路径:")
    print(f"查询文件: {query_file}")
    print(f"候选文件: {cand_file}")
    print(f"查询文件存在: {os.path.exists(query_file)}")
    print(f"候选文件存在: {os.path.exists(cand_file)}")
    
    if not os.path.exists(query_file) or not os.path.exists(cand_file):
        print("❌ 数据文件不存在")
        return False
    
    # 读取并检查数据
    print("\n检查查询数据...")
    queries = []
    with open(query_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 只检查前5条
                break
            try:
                query = json.loads(line.strip())
                queries.append(query)
                print(f"查询 {i+1}:")
                print(f"  - qid: {query.get('qid', 'None')}")
                print(f"  - query_txt: {repr(query.get('query_txt', 'None'))}")
                print(f"  - query_img_path: {repr(query.get('query_img_path', 'None'))}")
                print(f"  - query_modality: {query.get('query_modality', 'None')}")
            except Exception as e:
                print(f"解析查询 {i+1} 失败: {e}")
    
    print(f"\n检查候选数据...")
    candidates = []
    with open(cand_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 只检查前5条
                break
            try:
                candidate = json.loads(line.strip())
                candidates.append(candidate)
                print(f"候选 {i+1}:")
                print(f"  - did: {candidate.get('did', 'None')}")
                print(f"  - txt: {repr(candidate.get('txt', 'None'))}")
                print(f"  - img_path: {repr(candidate.get('img_path', 'None'))}")
                print(f"  - modality: {candidate.get('modality', 'None')}")
            except Exception as e:
                print(f"解析候选 {i+1} 失败: {e}")
    
    print(f"\n✅ 数据加载测试完成")
    print(f"查询样本数: {len(queries)}")
    print(f"候选样本数: {len(candidates)}")
    
    return True

if __name__ == "__main__":
    test_data_loading()
