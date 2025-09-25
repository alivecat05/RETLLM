#!/usr/bin/env python3
"""
简化版CLIP评估脚本
支持CLIP和Long-CLIP模型的嵌入提取和检索评估
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import json
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

class SimpleCLIPEvaluator:
    """简化的CLIP评估器"""
    
    def __init__(self, clip_type: str = "CLIP", device: str = "cuda:0"):
        self.clip_type = clip_type
        self.device = device
        self.clip_model = None
        self.preprocess = None
        self.tokenizer = None
        self._init_models()
    
    def _init_models(self):
        """初始化CLIP模型"""
        print(f"初始化 {self.clip_type} 模型...")
        
        if self.clip_type == "Long-CLIP":
            sys.path.insert(0, '/root/dws/MCS/Long-CLIP')
            sys.path.insert(0, '/root/dws/MCS/Long-CLIP/model')
            from model import longclip as clip
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
        
        print(f"{self.clip_type} 模型加载完成")
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """批量编码文本"""
        if not texts or all(not text.strip() for text in texts):
            return torch.zeros(len(texts), 768).to(self.device)
        
        # 过滤空文本
        valid_texts = [text.strip() if text.strip() else " " for text in texts]
        
        text_tokens = self.tokenizer(valid_texts, truncate=True).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens).float()
        return F.normalize(text_features, dim=-1)
    
    @torch.no_grad()
    def encode_images(self, image_paths: List[str], img_dir: str = "") -> torch.Tensor:
        """批量编码图像"""
        features = []
        
        for img_path in image_paths:
            if not img_path or img_path.strip() == "":
                # 空图像，使用零向量
                features.append(torch.zeros(1, 768).to(self.device))
                continue
            
            full_path = os.path.join(img_dir, img_path) if img_dir else img_path
            
            try:
                image = Image.open(full_path).convert('RGB')
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.clip_model.encode_image(image_input).float()
                features.append(F.normalize(image_features, dim=-1))
            except Exception as e:
                print(f"无法加载图像 {full_path}: {e}")
                features.append(torch.zeros(1, 768).to(self.device))
        
        return torch.cat(features, dim=0)
    
    def encode_multimodal(self, texts: List[str], image_paths: List[str], 
                         img_dir: str = "", fusion_method: str = "average") -> torch.Tensor:
        """编码多模态数据"""
        text_features = self.encode_text(texts)
        image_features = self.encode_images(image_paths, img_dir)
        
        if fusion_method == "average":
            # 平均融合
            combined = (text_features + image_features) / 2
        elif fusion_method == "text_only":
            combined = text_features
        elif fusion_method == "image_only":
            combined = image_features
        else:
            # 默认平均融合
            combined = (text_features + image_features) / 2
        
        return F.normalize(combined, dim=-1)

def batch_encode_data(evaluator: SimpleCLIPEvaluator, 
                     data: List[Dict], 
                     img_dir: str = "",
                     batch_size: int = 32,
                     mode: str = "multimodal") -> torch.Tensor:
    """批量编码数据"""
    all_features = []
    
    for i in tqdm(range(0, len(data), batch_size), desc=f"编码{mode}数据"):
        batch = data[i:i+batch_size]
        
        if mode == "multimodal":
            texts = [item.get('text', '') for item in batch]
            images = [item.get('image', '') for item in batch]
            features = evaluator.encode_multimodal(texts, images, img_dir)
        elif mode == "text":
            texts = [item.get('text', '') for item in batch]
            features = evaluator.encode_text(texts)
        elif mode == "image":
            images = [item.get('image', '') for item in batch]
            features = evaluator.encode_images(images, img_dir)
        
        all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0)

def compute_retrieval_metrics(query_features: torch.Tensor,
                            candidate_features: torch.Tensor,
                            ground_truth_indices: List[int],
                            k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """计算检索指标"""
    # 计算相似度矩阵
    similarity = torch.cosine_similarity(
        query_features.unsqueeze(1), 
        candidate_features.unsqueeze(0), 
        dim=-1
    )
    
    results = {}
    
    for k in k_values:
        correct = 0
        total = len(ground_truth_indices)
        
        for i, gt_idx in enumerate(ground_truth_indices):
            # 获取top-k
            _, top_k_indices = torch.topk(similarity[i], k=min(k, similarity.shape[1]))
            
            # 检查正确答案是否在top-k中
            if gt_idx in top_k_indices:
                correct += 1
        
        recall_at_k = correct / total if total > 0 else 0.0
        results[f'Recall@{k}'] = recall_at_k
    
    return results

def save_embeddings(embeddings: torch.Tensor, metadata: List[Dict], save_path: str):
    """保存嵌入和元数据"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'metadata': metadata}, f)
    print(f"嵌入已保存到: {save_path}, 形状: {embeddings.shape}")

def load_embeddings(load_path: str) -> Tuple[torch.Tensor, List[Dict]]:
    """加载嵌入和元数据"""
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    embeddings = data['embeddings']
    metadata = data['metadata']
    print(f"从 {load_path} 加载嵌入, 形状: {embeddings.shape}")
    return embeddings, metadata

def run_evaluation(config: Dict):
    """运行评估"""
    # 初始化评估器
    evaluator = SimpleCLIPEvaluator(
        clip_type=config['clip_type'], 
        device=config['device']
    )
    
    # 创建输出目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = output_dir / f"evaluation_{config['clip_type']}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"开始评估，配置: {config}")
    
    # 加载数据
    query_data = config['query_data']  # List[Dict] with 'text', 'image' keys
    candidate_data = config['candidate_data']  # List[Dict] with 'text', 'image' keys
    ground_truth = config['ground_truth']  # List[int]
    
    # 嵌入文件路径
    query_emb_path = output_dir / f"query_embeddings_{config['clip_type']}.pkl"
    candidate_emb_path = output_dir / f"candidate_embeddings_{config['clip_type']}.pkl"
    
    # 提取或加载查询嵌入
    if config.get('force_recompute', False) or not query_emb_path.exists():
        logging.info("提取查询嵌入...")
        query_embeddings = batch_encode_data(
            evaluator, query_data, config['img_dir'], 
            config['batch_size'], config['encoding_mode']
        )
        save_embeddings(query_embeddings, query_data, str(query_emb_path))
    else:
        logging.info("加载已有查询嵌入...")
        query_embeddings, _ = load_embeddings(str(query_emb_path))
    
    # 提取或加载候选嵌入
    if config.get('force_recompute', False) or not candidate_emb_path.exists():
        logging.info("提取候选嵌入...")
        candidate_embeddings = batch_encode_data(
            evaluator, candidate_data, config['img_dir'], 
            config['batch_size'], config['encoding_mode']
        )
        save_embeddings(candidate_embeddings, candidate_data, str(candidate_emb_path))
    else:
        logging.info("加载已有候选嵌入...")
        candidate_embeddings, _ = load_embeddings(str(candidate_emb_path))
    
    # 计算检索指标
    logging.info("计算检索指标...")
    results = compute_retrieval_metrics(
        query_embeddings, candidate_embeddings, 
        ground_truth, config['k_values']
    )
    
    # 保存结果
    results_file = output_dir / f"results_{config['clip_type']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印结果
    print(f"\n{config['clip_type']} 评估结果:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        logging.info(f"{metric}: {value:.4f}")
    
    logging.info(f"结果已保存到: {results_file}")
    
    return results

def main():
    """示例用法"""
    # 示例配置
    config = {
        'clip_type': 'CLIP',  # 或 'Long-CLIP'
        'device': 'cuda:0',
        'img_dir': '/path/to/images',
        'output_dir': '/root/dws/MCS/Codes_MBEIR/results/example',
        'batch_size': 32,
        'encoding_mode': 'multimodal',  # 'multimodal', 'text', 'image'
        'k_values': [1, 5, 10],
        'force_recompute': False,
        
        # 示例数据（需要根据实际数据调整）
        'query_data': [
            {'text': 'a cat sitting on a chair', 'image': 'cat.jpg'},
            {'text': 'a dog running in the park', 'image': 'dog.jpg'},
        ],
        'candidate_data': [
            {'text': 'cat on furniture', 'image': 'chair_cat.jpg'},
            {'text': 'dog outdoor activity', 'image': 'park_dog.jpg'},
            {'text': 'bird flying', 'image': 'bird.jpg'},
        ],
        'ground_truth': [0, 1]  # 第一个查询对应第一个候选，第二个查询对应第二个候选
    }
    
    # 运行评估
    results = run_evaluation(config)
    
    print("评估完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP检索评估")
    parser.add_argument('--clip_type', type=str, default='CLIP', 
                       choices=['CLIP', 'Long-CLIP'], help='CLIP模型类型')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--output_dir', type=str, 
                       default='/root/dws/MCS/Codes_MBEIR/results', help='输出目录')
    parser.add_argument('--force_recompute', action='store_true', 
                       help='强制重新计算嵌入')
    
    args = parser.parse_args()
    
    # 这里可以根据命令行参数调整配置
    print(f"使用参数: {args}")
    
    # 运行主函数
    main()
