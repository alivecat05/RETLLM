import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import gc
import json
import numpy as np
from PIL import Image
import logging
import time
from typing import Dict, List, Union, Tuple
from datasets import load_dataset

# 设置内存优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class MBEIRCLIPEvaluator:
    """M-BEIR数据集的CLIP评估器"""
    
    def __init__(self, clip_type: str = "CLIP", device: str = "cuda"):
        """
        初始化M-BEIR CLIP评估器
        
        Args:
            clip_type: "CLIP" 或 "Long-CLIP"
            device: 设备名称
        """
        self.clip_type = clip_type
        self.device = device
        self.clip_model = None
        self.preprocess = None
        self.tokenizer = None
        
        self._init_clip_model()
    
    def _init_clip_model(self):
        """初始化CLIP模型"""
        print(f"正在初始化 {self.clip_type} 模型...")

        self.clear_cache()
        
        # 加载SEARLE模型
        clip_model_name = "ViT-L/14"
        self.searle_model, self.encode_with_pseudo_tokens = torch.hub.load(
            repo_or_dir='/root/dws/MCS/SEARLE', 
            source='local', 
            model='searle',
            backbone=clip_model_name
        )
        self.searle_model.to(self.device)
        if self.clip_type == "Long-CLIP":
            sys.path.insert(0, '/root/dws/MCS/Long-CLIP')
            sys.path.insert(0, '/root/dws/MCS/Long-CLIP/model')
            from model import longclip as clip # type: ignore
            self.clip_model, self.preprocess = clip.load(
                "/root/dws/MCS/Long-CLIP/checkpoints/longclip-L.pt",
                device=self.device
            )
            self.tokenizer = clip.tokenize
            
        elif self.clip_type == "CLIP":
            sys.path.insert(0, '/root/dws/MCS/CLIP')
            import clip
            self.clip_model, self.preprocess = clip.load(
                "ViT-L/14@336px",
                device=self.device
            )
            self.tokenizer = clip.tokenize
        
        print(f"{self.clip_type} 模型初始化完成，使用设备: {self.device}")
        self.clear_cache()
    
    def clear_cache(self):
        """清理GPU和系统缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def extract_query_features(self, query_text: str, query_image_path: str = None, 
                              img_dir: str = None) -> torch.Tensor:
        """
        提取查询特征
        
        Args:
            query_text: 查询文本
            query_image_path: 查询图像路径
            img_dir: 图像目录
            
        Returns:
            归一化的查询特征向量
        """
        with torch.no_grad():
            features = []
            if query_text and query_text.strip() and query_image_path and query_image_path.strip() and img_dir:
                # 处理图像路径
                img_path = os.path.join(img_dir, query_image_path)
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                        image_features = self.clip_model.encode_image(image_input).float()
                        pseudo_tokens = self.searle_model(image_features.to(self.device))
                    except Exception as e:
                        print(f"图像加载失败: {img_path}, 错误: {e}")
                text_input = self.tokenizer([query_text.strip()], truncate=True).to(self.device)
                
                query_feature = self.encode_with_pseudo_tokens(self.clip_model, text_input, pseudo_tokens)
                features.append(F.normalize(query_feature, dim=-1))
            # 处理文本
            elif query_text and query_text.strip():
                text_input = self.tokenizer([query_text.strip()], truncate=True).to(self.device)
                text_features = self.clip_model.encode_text(text_input).float()
                features.append(F.normalize(text_features, dim=-1))
            
            # 处理图像
            elif query_image_path and query_image_path.strip() and img_dir:
                img_path = os.path.join(img_dir, query_image_path)
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                        image_features = self.clip_model.encode_image(image_input).float()
                        features.append(F.normalize(image_features, dim=-1))
                    except Exception as e:
                        print(f"图像加载失败: {img_path}, 错误: {e}")
            
            # 合并特征
            # if len(features) == 2:
            #     # 图像+文本：取平均
            #     combined_features = features[0] + features[1]
            #     return F.normalize(combined_features, dim=-1)
            # elif len(features) == 1:
            #     return features[0]
            # else:
            #     # 返回零向量
            #     feature_dim = 768
            #     return torch.zeros(1, feature_dim).to(self.device)
            if len(features) == 1:
                return features[0]
            else:
                # 返回零向量
                feature_dim = 768
                return torch.zeros(1, feature_dim).to(self.device)
    
    def extract_candidate_features(self, candidate_text: str, candidate_image_path: str = None,
                                 img_dir: str = None) -> torch.Tensor:
        """
        提取候选特征
        
        Args:
            candidate_text: 候选文本
            candidate_image_path: 候选图像路径
            img_dir: 图像目录
            
        Returns:
            归一化的候选特征向量
        """
        return self.extract_query_features(candidate_text, candidate_image_path, img_dir)
    
    def extract_batch_features(self, items: List[Dict], img_dir: str, 
                              text_key: str, image_key: str, 
                              batch_size: int = 32) -> torch.Tensor:

        all_features = []
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(items), batch_size), desc="批处理提取特征"):
                batch_end = min(batch_start + batch_size, len(items))
                batch_items = items[batch_start:batch_end]
                
                # 分离文本和图像数据
                batch_texts = []
                batch_images = []
                batch_has_text = []
                batch_has_image = []
                
                for item in batch_items:
                    text = item.get(text_key, '')
                    image_path = item.get(image_key, '')
                    
                    # 处理文本
                    if text and text.strip():
                        batch_texts.append(text.strip())
                        batch_has_text.append(True)
                    else:
                        batch_texts.append('')
                        batch_has_text.append(False)
                    
                    # 处理图像
                    has_valid_image = False
                    if image_path and image_path.strip() and img_dir:
                        img_path = os.path.join(img_dir, image_path)
                        if os.path.exists(img_path):
                            try:
                                image = Image.open(img_path).convert('RGB')
                                batch_images.append(image)
                                has_valid_image = True
                            except Exception as e:
                                print(f"图像加载失败: {img_path}, 错误: {e}")
                    
                    batch_has_image.append(has_valid_image)
                    if not has_valid_image:
                        batch_images.append(None)
                
                # 批量编码文本
                text_features_batch = None
                if any(batch_has_text):
                    valid_texts = [text for text, has_text in zip(batch_texts, batch_has_text) if has_text]
                    if valid_texts:
                        text_tokens = self.tokenizer(valid_texts, truncate=True).to(self.device)
                        text_features_batch = self.clip_model.encode_text(text_tokens).float()
                        text_features_batch = F.normalize(text_features_batch, dim=-1)
                
                # 批量编码图像
                image_features_batch = None
                if any(batch_has_image):
                    valid_images = [img for img, has_img in zip(batch_images, batch_has_image) if has_img]
                    if valid_images:
                        image_inputs = torch.stack([self.preprocess(img) for img in valid_images]).to(self.device)
                        image_features_batch = self.clip_model.encode_image(image_inputs).float()
                        image_features_batch = F.normalize(image_features_batch, dim=-1)
                
                # 合并特征
                batch_features = []
                text_idx = 0
                image_idx = 0
                
                for i in range(len(batch_items)):
                    features = []
                    
                    # 添加文本特征
                    if batch_has_text[i] and text_features_batch is not None:
                        features.append(text_features_batch[text_idx:text_idx+1])
                        text_idx += 1
                    
                    # 添加图像特征
                    if batch_has_image[i] and image_features_batch is not None:
                        features.append(image_features_batch[image_idx:image_idx+1])
                        image_idx += 1
                    
                    # 合并特征
                    if len(features) == 2:
                        # 多模态：相加后归一化
                        # combined = features[0] + features[1]
                        combined = features[1]
                        final_feature = F.normalize(combined, dim=-1)
                    elif len(features) == 1:
                        final_feature = features[0]
                    else:
                        # 零向量
                        feature_dim = 768  # 根据模型调整
                        final_feature = torch.zeros(1, feature_dim).to(self.device)
                    
                    batch_features.append(final_feature.cpu().float())
                
                all_features.extend(batch_features)
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return torch.cat(all_features, dim=0)

def load_mbeir_data(data_dir: str, task_name: str, split: str = "test"):
    """
    加载M-BEIR数据集
    
    Args:
        data_dir: M-BEIR数据目录路径
        task_name: 任务名称，如 "mbeir_mscoco_task0"
        split: 数据分割，"test", "val" 等
    
    Returns:
        queries, candidates
    """
    # 查询文件路径
    query_file = os.path.join(data_dir, "query", split, f"{task_name}_{split}.jsonl")
    
    # 候选池文件路径
    if "mscoco" in task_name:
        cand_file = os.path.join(data_dir, "cand_pool", "local", f"{task_name}_{split}_cand_pool.jsonl")
    else:
        cand_file = os.path.join(data_dir, "cand_pool", "local", f"{task_name}_cand_pool.jsonl")
    
    # 读取查询数据
    queries = []
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    
    # 读取候选数据
    candidates = []
    with open(cand_file, 'r', encoding='utf-8') as f:
        for line in f:
            candidates.append(json.loads(line.strip()))
    
    print(f"加载完成: {len(queries)} 个查询, {len(candidates)} 个候选")
    return queries, candidates

def extract_query_embeddings(evaluator: MBEIRCLIPEvaluator, 
                           queries: List[Dict], 
                           img_dir: str,
                           save_path: str,
                           batch_size: int = 32) -> Tuple[torch.Tensor, List]:
    """提取查询嵌入 - 批处理版本"""
    print(f"开始批处理提取 {len(queries)} 个查询的嵌入 (批大小: {batch_size})")
    
    # 使用批处理提取特征
    embeddings = evaluator.extract_batch_features(
        items=queries,
        img_dir=img_dir,
        text_key='query_txt',
        image_key='query_img_path',
        batch_size=batch_size
    )
    
    # 构建元数据
    metadata = []
    for item in queries:
        metadata.append({
            'qid': item.get('qid', ''),
            'query_txt': item.get('query_txt', ''),
            'query_img_path': item.get('query_img_path', ''),
            'query_modality': item.get('query_modality', ''),
            'task_id': item.get('task_id', ''),
            'pos_cand_list': item.get('pos_cand_list', []),
            'neg_cand_list': item.get('neg_cand_list', [])
        })
    
    # 保存嵌入
    with open(save_path, 'wb') as f:
        pickle.dump((embeddings, metadata), f)
    
    print(f"查询嵌入已保存到: {save_path}, 形状: {embeddings.shape}")
    return embeddings, metadata

def extract_candidate_embeddings(evaluator: MBEIRCLIPEvaluator,
                               candidates: List[Dict],
                               img_dir: str,
                               save_path: str,
                               batch_size: int = 32) -> Tuple[torch.Tensor, List]:
    """提取候选嵌入 - 批处理版本"""
    print(f"开始批处理提取 {len(candidates)} 个候选的嵌入 (批大小: {batch_size})")
    
    # 使用批处理提取特征
    embeddings = evaluator.extract_batch_features(
        items=candidates,
        img_dir=img_dir,
        text_key='txt',
        image_key='img_path',
        batch_size=batch_size
    )
    
    # 构建元数据
    metadata = []
    for idx, candidate in enumerate(candidates):
        metadata.append({
            'did': candidate.get('did', ''),
            'txt': candidate.get('txt', ''),
            'img_path': candidate.get('img_path', ''),
            'modality': candidate.get('modality', ''),
            'candidate_idx': idx
        })
    
    # 保存嵌入
    with open(save_path, 'wb') as f:
        pickle.dump((embeddings, metadata), f)
    
    print(f"候选嵌入已保存到: {save_path}, 形状: {embeddings.shape}")
    return embeddings, metadata


def evaluate_mbeir_retrieval(query_embeddings: torch.Tensor,
                           candidate_embeddings: torch.Tensor,
                           query_metadata: List,
                           candidate_metadata: List,
                           k_values: List[int] = [1, 5, 10, 20],
                           batch_size: int = 100) -> Dict[str, float]:

    print(f"查询数量: {len(query_embeddings)}, 候选数量: {len(candidate_embeddings)}")
    print(f"使用批处理大小: {batch_size}")
    
    # 预处理：创建候选ID数组和查询正确答案集合
    print("预处理数据...")
    candidate_dids = np.array([cand_meta['did'] for cand_meta in candidate_metadata])
    
    # 只处理有正确答案的查询
    valid_queries = []
    query_pos_sets = []
    for i, query_meta in enumerate(query_metadata):
        pos_list = query_meta.get('pos_cand_list', [])
        if pos_list:
            valid_queries.append(i)
            query_pos_sets.append(set(pos_list))
    
    print(f"有效查询数量: {len(valid_queries)}/{len(query_metadata)}")
    
    # 计算最大k值，一次性计算top-k
    max_k = max(k_values)
    max_k = min(max_k, len(candidate_embeddings))
    
    results = {}
    all_top_k_indices = []
    
    # 检查数据规模并选择计算策略
    num_queries = len(query_embeddings)
    num_candidates = len(candidate_embeddings)
    total_pairs = num_queries * num_candidates
    
    print("计算相似度和top-k索引...")
    print(f"总计算量: {num_queries} × {num_candidates} = {total_pairs:,} 个相似度值")
    
    # 记录开始时间
    start_time = time.time()
    
    # 优化的相似度计算：使用矩阵乘法代替cosine_similarity
    # 先归一化嵌入向量（如果还没有归一化）
    query_embeddings = F.normalize(query_embeddings, dim=-1)
    candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
    
    # 对于超大数据集，考虑使用候选分块策略
    use_candidate_chunking = total_pairs > 50000000  # 超过5千万对时启用候选分块
    candidate_chunk_size = 50000 if use_candidate_chunking else len(candidate_embeddings)
    
    if use_candidate_chunking:
        print(f"数据集过大，启用候选分块策略，候选分块大小: {candidate_chunk_size}")
    
    # 分批计算相似度和top-k索引（只计算一次）
    num_batches = (len(query_embeddings) + batch_size - 1) // batch_size
    for batch_idx, batch_start in enumerate(tqdm(range(0, len(query_embeddings), batch_size), 
                           desc="计算相似度矩阵")):
        batch_end = min(batch_start + batch_size, len(query_embeddings))
        batch_query_embeddings = query_embeddings[batch_start:batch_end]
        
        # 估算剩余时间
        if batch_idx > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / batch_idx
            remaining_batches = num_batches - batch_idx
            estimated_remaining_time = avg_time_per_batch * remaining_batches
            
        batch_end = min(batch_start + batch_size, len(query_embeddings))
        batch_query_embeddings = query_embeddings[batch_start:batch_end]
        
        if use_candidate_chunking:
            # 对候选也进行分块处理
            batch_similarities = []
            batch_indices = []
            
            for cand_start in range(0, len(candidate_embeddings), candidate_chunk_size):
                cand_end = min(cand_start + candidate_chunk_size, len(candidate_embeddings))
                cand_chunk = candidate_embeddings[cand_start:cand_end]
                
                with torch.no_grad():
                    # 使用矩阵乘法计算相似度（更高效）
                    chunk_similarity = torch.mm(batch_query_embeddings, cand_chunk.T)
                    batch_similarities.append(chunk_similarity.cpu())
                    
                    # 清理GPU内存
                    del chunk_similarity
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 合并所有候选块的结果，找到全局top-k
            with torch.no_grad():
                full_similarity = torch.cat(batch_similarities, dim=1)
                _, batch_top_k_indices = torch.topk(full_similarity, k=max_k, dim=-1)
                all_top_k_indices.append(batch_top_k_indices.numpy())
                
                # 清理内存
                del batch_similarities, full_similarity, batch_top_k_indices
                
        else:
            # 常规处理
            with torch.no_grad():
                # 使用矩阵乘法计算相似度（更高效）
                # 相似度 = 查询 @ 候选.T
                similarity_matrix = torch.mm(batch_query_embeddings, candidate_embeddings.T)
                
                # 一次性计算top-max_k索引
                _, batch_top_k_indices = torch.topk(similarity_matrix, k=max_k, dim=-1)
                all_top_k_indices.append(batch_top_k_indices.cpu().numpy())
                
                # 清理内存
                del similarity_matrix, batch_top_k_indices
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # 合并所有批次的top-k索引
    all_top_k_indices = np.concatenate(all_top_k_indices, axis=0)
    
    total_time = time.time() - start_time
    # print(f"相似度计算完成，总耗时: {total_time/60:.1f} 分钟")
    # print(f"平均每批次耗时: {total_time/num_batches:.2f} 秒")
    
    print("计算各k值的recall指标...")
    # 现在为每个k值计算recall（重用top-k索引）
    for k in k_values:
        current_k = min(k, max_k)
        correct = 0
        
        # 批量处理有效查询
        for idx in tqdm(range(len(valid_queries)), desc=f"计算Recall@{k}", leave=False):
            query_idx = valid_queries[idx]
            pos_set = query_pos_sets[idx]
            
            # 获取当前查询的top-k候选ID（使用预计算的索引）
            top_k_cand_indices = all_top_k_indices[query_idx, :current_k]
            top_k_cand_dids = candidate_dids[top_k_cand_indices]
            
            # 使用集合交集快速检查命中
            if pos_set.intersection(top_k_cand_dids):
                correct += 1
        
        total = len(valid_queries)
        recall_at_k = correct / total if total > 0 else 0.0
        results[f'Recall@{k}'] = recall_at_k
        print(f"Recall@{k}: {recall_at_k:.4f} ({correct}/{total})")
    
    return results

def get_topk_candidates_for_rerank(query_embeddings: torch.Tensor,
                                 candidate_embeddings: torch.Tensor,
                                 query_metadata: List,
                                 candidate_metadata: List,
                                 queries: List[Dict],
                                 candidates: List[Dict],
                                 img_dir: str,
                                 k: int = 5,
                                 batch_size: int = 100) -> List[Dict]:
    print(f"开始获取每个查询的top-{k}候选项...")
    
    # 归一化嵌入向量
    query_embeddings = F.normalize(query_embeddings, dim=-1)
    candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
    
    results = []
    
    # 分批处理查询
    num_batches = (len(query_embeddings) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="获取top-k候选"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(query_embeddings))
        batch_query_embeddings = query_embeddings[batch_start:batch_end]
        
        with torch.no_grad():
            # 计算相似度矩阵
            similarity_matrix = torch.mm(batch_query_embeddings, candidate_embeddings.T)
            
            # 获取top-k索引和分数
            similarities, batch_top_k_indices = torch.topk(similarity_matrix, k=k, dim=-1)
            batch_top_k_indices = batch_top_k_indices.cpu().numpy()
            similarities = similarities.cpu().numpy()
            
            # 处理当前批次的每个查询
            for i, query_idx in enumerate(range(batch_start, batch_end)):
                query_meta = query_metadata[query_idx]
                query_data = queries[query_idx]
                
                # 获取top-k候选索引和分数
                top_k_indices = batch_top_k_indices[i]
                top_k_scores = similarities[i]
                
                # 构建候选项列表
                top_k_candidates = []
                for rank, (cand_idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
                    cand_meta = candidate_metadata[cand_idx]
                    cand_data = candidates[cand_idx]
                    
                    # 构建候选项信息
                    candidate_info = {
                        'rank': rank + 1,
                        'candidate_idx': int(cand_idx),
                        'did': cand_meta['did'],
                        'txt': cand_meta['txt'],
                        'img_path': cand_meta['img_path'],
                        'modality': cand_meta['modality'],
                        'similarity_score': float(score),
                        # 添加完整图像路径
                        'full_img_path': os.path.join(img_dir, cand_meta['img_path']) if cand_meta['img_path'] else None
                    }
                    top_k_candidates.append(candidate_info)
                
                # 构建查询结果
                query_result = {
                    'query_idx': query_idx,
                    'qid': query_meta['qid'],
                    'query_txt': query_meta['query_txt'],
                    'query_img_path': query_meta['query_img_path'],
                    'query_modality': query_meta['query_modality'],
                    'full_query_img_path': os.path.join(img_dir, query_meta['query_img_path']) if query_meta['query_img_path'] else None,
                    'pos_cand_list': query_meta['pos_cand_list'],
                    'top_k_candidates': top_k_candidates
                }
                
                results.append(query_result)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"成功获取 {len(results)} 个查询的top-{k}候选项")
    return results

def save_topk_for_rerank(topk_results: List[Dict], save_path: str):
    """
    保存top-k结果用于rerank
    
    Args:
        topk_results: top-k结果列表
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(topk_results, f, indent=2, ensure_ascii=False)
    print(f"Top-k结果已保存到: {save_path}")

def load_topk_for_rerank(load_path: str) -> List[Dict]:
    """
    加载top-k结果用于rerank
    
    Args:
        load_path: 加载路径
        
    Returns:
        top-k结果列表
    """
    with open(load_path, 'r', encoding='utf-8') as f:
        topk_results = json.load(f)
    print(f"已从 {load_path} 加载 {len(topk_results)} 个查询的top-k结果")
    return topk_results

def calculate_rerank_accuracy(reranked_results: List[Dict], k_values: List[int] = [1]) -> Dict[str, float]:
    
    results = {}

    
    return results

def calculate_recall_from_results(results: List[Dict]) -> float:
    """从结果计算召回率"""
    total_samples = len(results)
    if total_samples == 0:
        return 0.0
    correct_count = sum(1 for res in results if res['is_correct'])
    return correct_count / total_samples

def evaluate_single_task(task_name: str, clip_type: str, device: str, 
                        mbeir_dir: str, img_dir: str, split: str = "test") -> Dict[str, float]:
    """评估单个任务"""
    print(f"\n{'='*50}")
    print(f"开始评估任务: {task_name}")
    print(f"CLIP模型: {clip_type}")
    print(f"{'='*50}")
    
    save_dir = f"/root/dws/MCS/Codes_MBEIR/embeddings/{task_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置任务特定的日志
    log_path = os.path.join(save_dir, f"{task_name}_{clip_type}_evaluation.log")
    
    # 创建任务特定的logger
    task_logger = logging.getLogger(task_name)
    task_logger.setLevel(logging.INFO)
    
    # 清除之前的handlers
    for handler in task_logger.handlers[:]:
        task_logger.removeHandler(handler)
    
    # 添加文件handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    task_logger.addHandler(file_handler)
    
    # 初始化评估器
    evaluator = MBEIRCLIPEvaluator(clip_type=clip_type, device=device)
    
    try:
        # 检查数据文件是否存在
        query_file = os.path.join(mbeir_dir, "query", split, f"{task_name}_{split}.jsonl")
        if "mscoco" in task_name:
            cand_file = os.path.join(mbeir_dir, "cand_pool", "local", f"{task_name}_{split}_cand_pool.jsonl")
        else:
            cand_file = os.path.join(mbeir_dir, "cand_pool", "local", f"{task_name}_cand_pool.jsonl")
        
        if not os.path.exists(query_file):
            print(f"❌ 查询文件不存在: {query_file}")
            task_logger.error(f"查询文件不存在: {query_file}")
            return {}
        
        if not os.path.exists(cand_file):
            print(f"❌ 候选文件不存在: {cand_file}")
            task_logger.error(f"候选文件不存在: {cand_file}")
            return {}
        
        # 加载M-BEIR数据集
        print(f"正在加载 {task_name} 数据集...")
        queries, candidates = load_mbeir_data(mbeir_dir, task_name, split)
        
        if len(queries) == 0 or len(candidates) == 0:
            print(f"❌ 数据集为空: queries={len(queries)}, candidates={len(candidates)}")
            task_logger.error(f"数据集为空: queries={len(queries)}, candidates={len(candidates)}")
            return {}
        
        # 嵌入文件路径
        query_save_path = os.path.join(save_dir, f"query_embeddings_{clip_type}.pkl")
        candidate_save_path = os.path.join(save_dir, f"candidate_embeddings_{clip_type}.pkl")
        
        # 动态调整批处理大小
        num_queries = len(queries)
        num_candidates = len(candidates)
        
        # 根据数据量调整嵌入提取的批处理大小
        if num_queries > 10000 or num_candidates > 50000:
            embedding_batch_size = 16  # 大数据集用小批次
        elif num_queries > 1000 or num_candidates > 10000:
            embedding_batch_size = 32  # 中等数据集
        else:
            embedding_batch_size = 64  # 小数据集用大批次
        
        print(f"使用嵌入提取批处理大小: {embedding_batch_size}")
        
        # 提取查询嵌入
        if not os.path.exists(query_save_path):
            print("正在批处理提取查询嵌入...")
            query_embeddings, query_metadata = extract_query_embeddings(
                evaluator, queries, img_dir, query_save_path, batch_size=64
            )
        else:
            print("正在加载已保存的查询嵌入...")
            with open(query_save_path, 'rb') as f:
                query_embeddings, query_metadata = pickle.load(f)
            print(f"从文件加载查询嵌入: {query_embeddings.shape}")
        
        # 提取候选嵌入
        if not os.path.exists(candidate_save_path):
            print("正在批处理提取候选嵌入...")
            candidate_embeddings, candidate_metadata = extract_candidate_embeddings(
                evaluator, candidates, img_dir, candidate_save_path, batch_size=embedding_batch_size
            )
        else:
            print("正在加载已保存的候选嵌入...")
            with open(candidate_save_path, 'rb') as f:
                candidate_embeddings, candidate_metadata = pickle.load(f)
            print(f"从文件加载候选嵌入: {candidate_embeddings.shape}")
        
        torch.cuda.empty_cache()
        print("开始评估检索性能...")
        query_embeddings = query_embeddings.cpu().detach()
        candidate_embeddings = candidate_embeddings.cpu().detach()
        
        # 根据数据大小动态调整批处理大小
        num_queries = len(query_embeddings)
        num_candidates = len(candidate_embeddings)
        
        # 更智能的批处理大小调整策略
        # 考虑GPU内存限制和计算效率
        total_pairs = num_queries * num_candidates
        
        if total_pairs > 100000000:  # 1亿对，超大数据集
            batch_size = 8
        elif total_pairs > 50000000:  # 5千万对，大数据集
            batch_size = 16
        elif total_pairs > 10000000:  # 1千万对，中大数据集
            batch_size = 32
        elif total_pairs > 1000000:   # 100万对，中等数据集
            batch_size = 64
        else:
            batch_size = 128  # 小数据集可以用更大批次
            
        print(f"检测到查询x候选={num_queries}x{num_candidates}={total_pairs:,}, 使用批处理大小: {batch_size}")
        
        results = evaluate_mbeir_retrieval(
            query_embeddings, 
            candidate_embeddings, 
            query_metadata,
            candidate_metadata,
            k_values=[1, 5, 10, 20],
            batch_size=batch_size
        )
        
        # 获取top-5候选项用于rerank
        print("正在获取top-5候选项用于rerank...")
        topk_results = get_topk_candidates_for_rerank(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            query_metadata=query_metadata,
            candidate_metadata=candidate_metadata,
            queries=queries,
            candidates=candidates,
            img_dir=img_dir,
            k=5,
            batch_size=batch_size
        )
        
        # 保存top-k结果
        topk_save_path = os.path.join(save_dir, f"topk_for_rerank_{clip_type}.json")
        save_topk_for_rerank(topk_results, topk_save_path)
        
        # 保存结果
        results_path = os.path.join(save_dir, f"evaluation_results_{clip_type}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ 任务 {task_name} 评估完成")
        print(f"结果已保存到: {results_path}")
        
        # 记录详细结果
        task_logger.info(f"任务: {task_name}")
        task_logger.info(f"数据分割: {split}")
        task_logger.info(f"CLIP模型: {clip_type}")
        task_logger.info(f"查询数量: {len(query_embeddings)}")
        task_logger.info(f"候选数量: {len(candidate_embeddings)}")
        
        for metric, value in results.items():
            task_logger.info(f"{metric}: {value:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ 任务 {task_name} 评估失败: {e}")
        task_logger.error(f"评估错误: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    finally:
        # 清理资源
        evaluator.clear_cache()
        # 关闭logger handlers
        for handler in task_logger.handlers[:]:
            handler.close()
            task_logger.removeHandler(handler)


def main():
    """主评估流程 - 自动遍历所有数据集"""
    # 配置参数
    device = "cuda:6"
    clip_type = "Long-CLIP"  # 或 "Long-CLIP"
    
    # M-BEIR数据集路径和任务配置
    mbeir_dir = "/root/dws/MCS/Datasets/M-BEIR"
    img_dir = "/root/dws/MCS/Datasets/M-BEIR"  # 图像基础目录

    available_tasks = [
 
        # "mbeir_cirr_task7",
        # "mbeir_fashioniq_task7",
        "mbeir_webqa_task2",
        # "mbeir_edis_task2",
        # "mbeir_infoseek_task6",
        # "mbeir_oven_task8",
        ""
    ]
    
    split = "test"
    
    # 创建总体结果保存目录
    overall_results_dir = "/root/dws/MCS/Codes_MBEIR/overall_results"
    os.makedirs(overall_results_dir, exist_ok=True)
    
    # 设置总体日志
    overall_log_path = os.path.join(overall_results_dir, f"overall_{clip_type}_evaluation.log")
    logging.basicConfig(
        filename=overall_log_path, 
        filemode='w', 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print(f"开始自动评估所有M-BEIR数据集")
    print(f"CLIP模型: {clip_type}")
    print(f"总共 {len(available_tasks)} 个任务")
    print(f"总体日志将保存到: {overall_log_path}")
    
    # 存储所有任务的结果
    all_results = {}
    successful_tasks = []
    failed_tasks = []
    
    # 遍历所有任务
    for i, task_name in enumerate(available_tasks, 1):
        print(f"\n进度: [{i}/{len(available_tasks)}]")
        
        try:
            # 评估单个任务
            task_results = evaluate_single_task(
                task_name=task_name,
                clip_type=clip_type,
                device=device,
                mbeir_dir=mbeir_dir,
                img_dir=img_dir,
                split=split
            )
            
            if task_results:
                all_results[task_name] = task_results
                successful_tasks.append(task_name)
                logging.info(f"任务 {task_name} 评估成功")
                
                # 打印简要结果
                print(f"任务 {task_name} 结果:")
                for metric, value in task_results.items():
                    print(f"  {metric}: {value:.4f}")
            else:
                failed_tasks.append(task_name)
                logging.error(f"任务 {task_name} 评估失败")
                
        except Exception as e:
            print(f"❌ 任务 {task_name} 出现异常: {e}")
            failed_tasks.append(task_name)
            logging.error(f"任务 {task_name} 异常: {e}")
    
    # 保存所有结果
    all_results_path = os.path.join(overall_results_dir, f"all_tasks_results_{clip_type}.json")
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 计算平均性能
    avg_results = {}
    avg_results_path = None
    if all_results:
        metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@20']
        
        for metric in metrics:
            values = [results[metric] for results in all_results.values() if metric in results]
            if values:
                avg_results[f"Avg_{metric}"] = sum(values) / len(values)
        
        # 保存平均结果
        if avg_results:
            avg_results_path = os.path.join(overall_results_dir, f"average_results_{clip_type}.json")
            with open(avg_results_path, 'w') as f:
                json.dump(avg_results, f, indent=2)
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"所有任务评估完成!")
    print(f"{'='*60}")
    print(f"成功完成: {len(successful_tasks)} 个任务")
    print(f"失败任务: {len(failed_tasks)} 个任务")
    
    if successful_tasks:
        print(f"\n✅ 成功的任务:")
        for task in successful_tasks:
            print(f"  - {task}")
    
    if failed_tasks:
        print(f"\n❌ 失败的任务:")
        for task in failed_tasks:
            print(f"  - {task}")
    
    if all_results:
        print(f"\n📊 平均性能:")
        for metric, value in avg_results.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\n📁 结果文件:")
    print(f"  - 所有结果: {all_results_path}")
    if avg_results_path:
        print(f"  - 平均结果: {avg_results_path}")
    print(f"  - 总体日志: {overall_log_path}")
    
    # 记录总结到日志
    logging.info(f"评估总结:")
    logging.info(f"成功任务: {len(successful_tasks)}")
    logging.info(f"失败任务: {len(failed_tasks)}")
    logging.info(f"成功任务列表: {successful_tasks}")
    logging.info(f"失败任务列表: {failed_tasks}")
    
    if avg_results:
        logging.info(f"平均性能:")
        for metric, value in avg_results.items():
            logging.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
