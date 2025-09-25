#!/usr/bin/env python3

import logging
import sys
import os
import json
from typing import List, Dict
sys.path.append('/root/dws/MCS/Codes')
from Qwen_vl import *
sys.path.append('/root/dws/MCS/Codes_MBEIR')
from mbeir_clip_eval import (
    MBEIRCLIPEvaluator,
    load_mbeir_data,
    get_topk_candidates_for_rerank,
    save_topk_for_rerank,

)
def find_candidate_index_by_did(candidates, target_did):
    for index, candidate in enumerate(candidates):
        if candidate['txt'] == target_did:
            return index
    return -1
def extract_topk_for_single_task(task_name: str, 
                                 clip_type: str = "Long-CLIP", 
                                 device: str = "cuda:6",
                                 k: int = 5):
    print(f"开始为任务 {task_name} 提取top-{k}候选项...")
    
    # 配置路径
    mbeir_dir = "/root/dws/MCS/Datasets/M-BEIR"
    img_dir = "/root/dws/MCS/Datasets/M-BEIR"
    save_dir = f"/root/dws/MCS/Codes_MBEIR/embeddings/{task_name}"
    
    # 检查嵌入文件是否存在
    query_save_path = os.path.join(save_dir, f"query_embeddings_{clip_type}.pkl")
    candidate_save_path = os.path.join(save_dir, f"candidate_embeddings_{clip_type}.pkl")
    
    if not os.path.exists(query_save_path) or not os.path.exists(candidate_save_path):
        print("❌ 嵌入文件不存在，请先运行主评估脚本生成嵌入")
        return None
    
    # 加载数据
    print("加载原始数据...")
    queries, candidates = load_mbeir_data(mbeir_dir, task_name, split="test")
    
    # 加载嵌入
    print("加载嵌入文件...")
    import pickle
    with open(query_save_path, 'rb') as f:
        query_embeddings, query_metadata = pickle.load(f)
    with open(candidate_save_path, 'rb') as f:
        candidate_embeddings, candidate_metadata = pickle.load(f)
    
    print(f"查询嵌入形状: {query_embeddings.shape}")
    print(f"候选嵌入形状: {candidate_embeddings.shape}")
    
    # 获取top-k候选项
    topk_results = get_topk_candidates_for_rerank(
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        query_metadata=query_metadata,
        candidate_metadata=candidate_metadata,
        queries=queries,
        candidates=candidates,
        img_dir=img_dir,
        k=k,
        batch_size=100
    )
    
    # 保存结果
    output_path = os.path.join(save_dir, f"topk_{k}_for_rerank_{clip_type}.json")
    save_topk_for_rerank(topk_results, output_path)
    
    
    print(f"✅ 成功提取 {len(topk_results)} 个查询的top-{k}候选项")
    print(f"文件保存位置:")
    print(f"  - Top-k原始数据: {output_path}")
    
    return topk_results
def show_topk_example(topk_results: List[Dict], num_examples: int = 2):
    """
    展示top-k结果的示例
    
    Args:
        topk_results: top-k结果
        num_examples: 展示的示例数量
    """
    print(f"\n{'='*60}")
    print(f"Top-K候选项示例（前{num_examples}个查询）")
    print(f"{'='*60}")
    
    for i, query_result in enumerate(topk_results[:num_examples]):
        print(f"\n查询 {i+1}:")
        print(f"  QID: {query_result['qid']}")
        print(f"  查询文本: {query_result['query_txt']}")
        print(f"  查询图像: {query_result['query_img_path']}")
        print(f"  查询模态: {query_result['query_modality']}")
        print(f"  正确答案: {query_result['pos_cand_list']}")
        
        print(f"  Top-{len(query_result['top_k_candidates'])}候选项:")
        for j, candidate in enumerate(query_result['top_k_candidates']):
            print(f"    Rank {candidate['rank']}: ")
            print(f"      DID: {candidate['did']}")
            print(f"      相似度: {candidate['similarity_score']:.4f}")
            print(f"      文本: {candidate['txt']}")
            print(f"      图像: {candidate['img_path']}")
            print(f"      模态: {candidate['modality']}")
            
            # 检查是否是正确答案
            is_correct = candidate['did'] in query_result['pos_cand_list']
            print(f"      ✅ 正确答案" if is_correct else "      ❌ 错误答案")
            print()
def calculate_recall_from_results(results: List[Dict], retrieval_type: str) -> float:
    """
    从重排序结果计算 Recall@1。
    """
    total_samples = len(results)
    correct_count = 0
    if total_samples == 0:
        return 0.0
    correct_count = sum(1 for res in results if res['is_correct'])
    recall = correct_count / total_samples
    return recall

def main():
    data_path = '/root/dws/MCS/Datasets/M-BEIR/'
    available_tasks = [
        "mbeir_infoseek_task6",
        "mbeir_cirr_task7",
        "mbeir_fashioniq_task7",
        "mbeir_webqa_task2",
        "mbeir_edis_task2",
        "mbeir_oven_task8"
    ]
    
    for task_name in available_tasks:
        # 创建独立的logger实例用于I2T检索
        i2t_logger = logging.getLogger(f'{task_name}')
        i2t_logger.setLevel(logging.INFO)
        
        # 清除之前的handlers
        for handler in i2t_logger.handlers[:]:
            i2t_logger.removeHandler(handler)
        
        # 创建文件handler
        i2t_handler = logging.FileHandler(f'/root/dws/MCS/Codes_MBEIR/logs/{task_name}_1.log', mode='w')
        i2t_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        i2t_handler.setFormatter(formatter)
        i2t_logger.addHandler(i2t_handler)
        print("步骤1: 提取top-k候选项...")
        i =0
        if task_name == "mbeir_infoseek_task6":
            k = 10
        else:
            k = 5
        topk_results = extract_topk_for_single_task(
            task_name=task_name,
            clip_type="Long-CLIP",
            device="cuda:6",
            k=k
        )

        if task_name in ["mbeir_cirr_task7","mbeir_fashioniq_task7","mbeir_webqa_task2","mbeir_edis_task2","mbeir_oven_task8"]:
            reranked_text_results = []

            for result in topk_results:
                query_text = result['query_txt'] if result['query_txt'] is not None else None
                query_image = Image.open(os.path.join(data_path,result['query_img_path'])) if result['query_img_path'] is not None else None # type: ignore
                top_k_samples = []
                if result['top_k_candidates'][0]['modality'] == 'image':
                    for can in result['top_k_candidates']:
                        image  = Image.open(os.path.join(data_path,can['img_path'])) # type: ignore
                        top_k_samples.append([image,None])
                elif result['top_k_candidates'][0]['modality'] == "image,text":
                    for can in result['top_k_candidates']:
                        image  = Image.open(os.path.join(data_path,can['img_path']))# type: ignore
                        top_k_samples.append([image,can['txt']])
                query = [query_text,query_image]
                
                best_index, predict, entropies_all = Qwen_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL = top_k_samples, captions = query) # type: ignore
                pred_idx = result['top_k_candidates'][best_index]['did']
                
                reranked_is_correct = pred_idx in result['pos_cand_list']
                reranked_text_results.append({
                    'is_correct': reranked_is_correct,
                })
                corrt  = result['pos_cand_list']
                accuracy = calculate_recall_from_results(reranked_text_results, retrieval_type='text')  
                i2t_logger.info(f'{i}|{len(topk_results)} ,image_retrieval_recall@1: {accuracy:.4f}')
                i2t_logger.info(f'Is_correct: {reranked_is_correct}')
                i2t_logger.info(f'Correct image index: {corrt}, Reasoned image index: {pred_idx}')
                i2t_logger.info(f'Scores:{predict}')
                i2t_logger.info(f'Entropy: {entropies_all}')
                i2t_logger.info('--------------------------------------')
                i+=1
        elif task_name in ["mbeir_infoseek_task6"]:
            reranked_text_results = []
            
            for i,result in enumerate(topk_results[4003:], start=4003):
                query_text = result['query_txt'] if result['query_txt'] is not None else None
                query_image = Image.open(os.path.join(data_path,result['query_img_path'])) if result['query_img_path'] is not None else None # type: ignore
                top_k_samples = []
                if result['top_k_candidates'][0]['modality'] == 'text':
                    for can in result['top_k_candidates']:
                        top_k_samples.append(can['txt'])
                query = [query_text,query_image]

                reasoned_pred = Qwen_fine_reasoner(image_path = query[1], top_k_labels = top_k_samples, is_saliency = False, query_text = query[0]) # type: ignore

                candidate_index = find_candidate_index_by_did(result['top_k_candidates'], reasoned_pred)
                if candidate_index != -1:
                    best_index = candidate_index
                else:
                    best_index = 0  
                pred_idx = result['top_k_candidates'][best_index]['did']
                
                reranked_is_correct = pred_idx in result['pos_cand_list']
                reranked_text_results.append({
                    'is_correct': reranked_is_correct,
                })
                corrt  = result['pos_cand_list']
                accuracy = calculate_recall_from_results(reranked_text_results, retrieval_type='text')  
                i2t_logger.info(f'{i}|{len(topk_results)} ,text_retrieval_recall@1: {accuracy:.4f}')
                i2t_logger.info(f'Is_correct: {reranked_is_correct}')
                i2t_logger.info(f'Correct image index: {corrt}, Reasoned image index: {pred_idx}')
                i2t_logger.info('--------------------------------------')
                
            




if __name__ == "__main__":
    main()
