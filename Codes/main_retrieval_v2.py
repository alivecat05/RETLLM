import logging

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
from Qwen_vl import *
# from vllm_qwen import Qwen_fine_reasoner_Retrieval_scores_vllm
import torch
from datasets import load_dataset
import os
from tqdm import tqdm
import json
import numpy as np

from typing import List, Dict   
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
torch.backends.cudnn.benchmark = True  # 启用cuDNN基准优化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def load_datasets(data_name,k):
    dataset = load_dataset(f'royokong/{data_name}_test', split='test')
    dataset = dataset.rename_column('text', 'caption')
    dataset = dataset.rename_column('image', 'img')
    if data_name == 'coco':
        dataset = dataset.map(lambda x: {'caption': x['caption'][:5]}, num_proc=4)
    dataset = dataset.select(range(k))
    
    return dataset

def I2T_fine_retrieval(texts_image_index,similarity_scores, datasets, k=5, clip_type='CLIPB',dataset_name='coco'):
    # 创建独立的logger实例用于I2T检索
    i2t_logger = logging.getLogger(f'I2T_{dataset_name}_{clip_type}')
    i2t_logger.setLevel(logging.INFO)
    
    # 清除之前的handlers
    for handler in i2t_logger.handlers[:]:
        i2t_logger.removeHandler(handler)
    
    # 创建文件handler
    i2t_handler = logging.FileHandler(f'/root/dws/MCS/logs/ablation/{dataset_name}_{clip_type}_i2t.log', mode='w')
    i2t_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    i2t_handler.setFormatter(formatter)
    i2t_logger.addHandler(i2t_handler)
    
    path = '/root/dws/MCS/Query_images'
    reranked_text_results = []
    num_images = similarity_scores.shape[1]
    os.makedirs(path, exist_ok=True)
    all_images =[]
    for img in tqdm(datasets['img']):
        all_images.append(img)
    all_captions = []
    all_texts = []
    for each in tqdm(datasets['caption'], desc="Processing captions"):
        all_texts.append(each)
        for i in range(len(each)):
            all_captions.append(each[i])
    

    for i in tqdm(range(num_images)):
        
        topk_texts = torch.topk(similarity_scores.T[i], k).indices.cpu().numpy()
        correct_text_indices = [j for j in range(i * 5, i * 5 + 5)]
        image_sample = all_images[i]
        topk_text_samples = [all_texts[idx // 5][idx % 5] for idx in topk_texts]
        image_sample.save(os.path.join(path, f'{i}.jpg'))
        # text_retrieval_results.append({
        #     'is_correct': topk_texts[0] in correct_text_indices
        # })
        reasoned_pred =Qwen_fine_reasoner_Retrieval_v2(I2T=True, image_PIL=[image_sample], captions=topk_text_samples)
        #reasoned_pred,logits =Qwen_fine_reasoner_Retrieval_v5(I2T=True, image_PIL=[image_sample], captions=topk_text_samples)
        try:
            #reasoned_pred_index = topk_texts[int(reasoned_pred)] if reasoned_pred is not None else topk_texts[0]
             reasoned_pred_index = all_captions.index(reasoned_pred) if reasoned_pred is not None else topk_texts[0]
            
        except:
            reasoned_pred = topk_texts[0]
        reranked_is_correct = reasoned_pred_index in correct_text_indices
        reranked_text_results.append({
            'is_correct': reranked_is_correct,
        })
        accuracy = calculate_recall_from_results(reranked_text_results, retrieval_type='text')
        if i%10==0:
            print(f'{i}|{num_images} ,image_retrieval_recall@1: {accuracy:.4f}')
        # if step%10==0:
        #     print(f'{j} ,image_retrieval_recall@1: {accuracy:.4f}')

        i2t_logger.info(f'{i}|{num_images} ,image_retrieval_recall@1: {accuracy:.4f}')
        i2t_logger.info(f'Is_correct: {reranked_is_correct}')
        i2t_logger.info(f'Correct image index: {correct_text_indices}, Reasoned image index: {reasoned_pred_index}')
        i2t_logger.info(f'Topk text indices: {topk_texts}')
        i2t_logger.info(f'Topk text sample: {topk_text_samples}')
        # i2t_logger.info(f'Topk text logits: {logits}')
        i2t_logger.info('--------------------------------------')
        
    accuracy = calculate_recall_from_results(reranked_text_results, retrieval_type='text')
    
    # 关闭并移除handler，确保日志文件被正确关闭
    i2t_handler.close()
    i2t_logger.removeHandler(i2t_handler)
    
    return accuracy
        


def T2I_fine_retrieval(texts_image_index,similarity_scores, datasets, k=5, clip_type='CLIPB',dataset_name='coco'):
    # 创建独立的logger实例用于T2I检索
    t2i_logger = logging.getLogger(f'T2I_{dataset_name}_{clip_type}')
    t2i_logger.setLevel(logging.INFO)
    
    # 清除之前的handlers
    for handler in t2i_logger.handlers[:]:
        t2i_logger.removeHandler(handler)
    
    # 创建文件handler
    t2i_handler = logging.FileHandler(f'/root/dws/MCS/logs/ablation/{dataset_name}_{clip_type}_t2i.log', mode='w')
    t2i_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    t2i_handler.setFormatter(formatter)
    t2i_logger.addHandler(t2i_handler)
    
    reranked_image_results = []
    num_texts = similarity_scores.shape[0]
    all_images = []
    for img in tqdm(datasets['img']):
        all_images.append(img)
    step = 0
    # for j in tqdm(hard_negative['groups']):
    for j in tqdm(range(num_texts)):     
        # path = f'/root/dws/MCS/Datasets/625_coco_images/Group{j}'
        topk_images = torch.topk(similarity_scores[j], k).indices.cpu()
        correct_image_index = texts_image_index[j]
        text_sample = datasets['caption'][j // 5][j % 5]
        topk_image_samples = [all_images[idx] for idx in topk_images]
        # reasoned_pred =Qwen_fine_reasoner_Retrieval_setwise_v2(image_PIL=topk_image_samples, captions=text_sample)
        
        # image_path = [os.path.join(path,f'{i}.jpg') for i in range(5)]
        # topk_image_samples = [Image.open(image) for image in image_path]
        
        reasoned_pred,logits,entropies_all=Qwen_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL = topk_image_samples,captions=text_sample, is_saliency=False,I2T = False,T2I = True)
        # reasoned_pred = Qwen_forward(captions=text_sample, image_PIL=topk_image_samples)
        reasoned_pred_index = topk_images[int(reasoned_pred)] if reasoned_pred is not None else topk_images[0]
        reranked_is_correct = correct_image_index in reasoned_pred_index
        reranked_image_results.append({
                    'is_correct': reranked_is_correct,
                })
        accuracy = calculate_recall_from_results(reranked_image_results, retrieval_type='image')
        if j%10==0:
            print(f'{j}|{num_texts} ,image_retrieval_recall@1: {accuracy:.4f}')
        t2i_logger.info(f'{j}|{num_texts} ,image_retrieval_recall@1: {accuracy:.4f}')
        t2i_logger.info(f'Is_correct: {reranked_is_correct}')
        t2i_logger.info(f'Correct image index: {correct_image_index}, Reasoned image index: {reasoned_pred_index}')
        t2i_logger.info(f'Text sample: {text_sample}')
        t2i_logger.info(f'Topk image indices: {topk_images}')
        t2i_logger.info(f'TopK image logits :{logits}')
        t2i_logger.info(f'TopK image entropies: {entropies_all}')
        t2i_logger.info('--------------------------------------')
        step+=1
    accuracy = calculate_recall_from_results(reranked_image_results, retrieval_type='image')
    
    # 关闭并移除handler，确保日志文件被正确关闭
    t2i_handler.close()
    t2i_logger.removeHandler(t2i_handler)
    
    return accuracy
def main(dataset_name,k,clip_type):
    path = '/root/dws/MCS/Datasets'
    datasets = load_datasets(dataset_name,k)   
    print(f'Loaded {dataset_name} dataset with {len(datasets)} samples.')
    with open(f'{path}/{dataset_name}_{clip_type}.json', 'r') as f:
        data = json.load(f)
        img_embs = torch.tensor(data['img_embs'])
        text_embs = torch.tensor(data['text_embs'])
        print(f'Loaded {dataset_name} embeddings from json file.')

    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
    scores = text_embs @ img_embs.T
    
    texts_image_index = [i // 5 for i in range(img_embs.shape[0]*5)]
    assert len(texts_image_index) == len(text_embs)
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    recall_k_list = [1, 5,10]
    batch_size = 64
    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()
    print(metrics)

    I2T_accuracy = I2T_fine_retrieval(texts_image_index,scores,datasets, k=5, clip_type=clip_,dataset_name=dataset_name)
    print(f'Final I2T accuracy: {I2T_accuracy:.3f}')
    T2I_accuracy = T2I_fine_retrieval(texts_image_index,scores,datasets, k=5, clip_type=clip_, dataset_name=dataset_name)

    print(f'Final T2I accuracy: {T2I_accuracy:.4f}')

if __name__ == "__main__":
    Multi_layers_fusion(
                self=model,
                starting_layer=5,
                ending_layer=16,
                entropy_threshold=0.75,
                mission = 'multi_text',#multi_image
                retracing_ratio=0.12,#0.05-0.35
                # retracing_ratio=0.12,#0.05-0.35
                vision_retracing_method = 'default',#default不使用视觉插入 adapt使用视觉插入
            )
    clip_type = ["Long-CLIP", "CLIPB"]  # 或 "Long-CLIP", "CLIPL14", "CLIPB"
    #clip_type = ["CLIP336"]  # 或 "Long-CLIP", "CLIPL14", "CLIPB", "CLIP336"
    dataset_name = ['flickr30k','coco']
    for dataset in dataset_name:
        if dataset == 'coco':
            k = 5000
        elif dataset == 'flickr30k':
            k = 1000
        for clip_ in clip_type:
            print(f"Processing dataset: {dataset} with CLIP type: {clip_}")

            main(dataset,k,clip_)
        
