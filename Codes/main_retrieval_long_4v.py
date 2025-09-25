import logging

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
from Qwen_vl import *
from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data as data
import os
from tqdm import tqdm
import json
import numpy as np
from typing import List, Dict   
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
torch.backends.cudnn.benchmark = True  # 启用cuDNN基准优化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data4v_root = '/root/dws/MCS/Datasets/ShareGPT4V/data/'
json_name = 'share-captioner_coco_lcs_sam_1246k_1107_cleaned.json'
image_root = '/root/dws/MCS/Datasets/ShareGPT4V/data/'

class share4v_val_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[:self.total_len]
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name).convert('RGB')
        return image, caption

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

def my_collate_fn(batch):
    images, captions = zip(*batch)

    return images, captions


def I2T_fine_retrieval(texts_image_index,similarity_scores, k=5):
    logging.basicConfig(filename=f'/root/dws/MCS/logs/ShareGPT4V_i2t.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    dataset = share4v_val_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn,num_workers=0)
    reranked_text_results = []
    num_images = similarity_scores.shape[1]
    all_images = []
    all_texts = []
    with torch.no_grad():
        for images, texts in dataloader:
            all_images.append(images[0])
            all_texts.extend(texts)
    step = 0

    for i in tqdm(range(num_images)):
        
        topk_texts = torch.topk(similarity_scores.T[i], k).indices.cpu().numpy()
        correct_text_indices = texts_image_index[i]
        image_sample = all_images[i]
        topk_text_samples = [all_texts[idx] for idx in topk_texts]
        reasoned_pred =Qwen_fine_reasoner_Retrieval_v2(I2T=True, image_PIL=[image_sample], captions=topk_text_samples,is_saliency=False)
        try:
            #reasoned_pred_index = topk_texts[int(reasoned_pred)] if reasoned_pred is not None else topk_texts[0]
             reasoned_pred_index = all_texts.index(reasoned_pred) if reasoned_pred is not None else topk_texts[0]
            
        except:
            reasoned_pred = topk_texts[0]
        reranked_is_correct = reasoned_pred_index in [correct_text_indices]
        reranked_text_results.append({
            'is_correct': reranked_is_correct,
        })
        accuracy = calculate_recall_from_results(reranked_text_results, retrieval_type='text')
        if i%10==0:
            print(f'{i}|{num_images} ,text_retrieval_recall@1: {accuracy:.4f}')
        # if step%10==0:
        #     print(f'{j} ,image_retrieval_recall@1: {accuracy:.4f}')

        logging.info(f'{i}|{num_images} ,text_retrieval_recall@1: {accuracy:.4f}')
        logging.info(f'Is_correct: {reranked_is_correct}')
        logging.info(f'Correct text index: {correct_text_indices}, Reasoned text index: {reasoned_pred_index}')
        logging.info(f'Topk text indices: {topk_texts}')
        logging.info(f'Topk text sample: {topk_text_samples}')
        # logging.info(f'Topk text logits: {logits}')
        logging.info('--------------------------------------')
        
    accuracy = calculate_recall_from_results(reranked_text_results, retrieval_type='text')
    return accuracy
        


def T2I_fine_retrieval(texts_image_index,similarity_scores, k=5):
    logging.basicConfig(filename=f'/root/dws/MCS/logs/ShareGPT4V_t2i.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    dataset = share4v_val_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn,num_workers=0)
    reranked_image_results = []
    num_texts = similarity_scores.shape[0]
    all_images = []
    all_texts = []
    with torch.no_grad():
        for images, texts in dataloader:
            all_images.append(images[0])
            all_texts.extend(texts)
    step = 0
 
    for j in tqdm(range(num_texts)):     
        topk_images = torch.topk(similarity_scores[j], k).indices.cpu()
        correct_image_index = texts_image_index[j]
        text_sample = all_texts[j]
        topk_image_samples = [all_images[idx] for idx in topk_images]

        
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
        logging.info(f'{j}|{num_texts} ,image_retrieval_recall@1: {accuracy:.4f}')
        logging.info(f'Is_correct: {reranked_is_correct}')
        logging.info(f'Correct image index: {correct_image_index}, Reasoned image index: {reasoned_pred_index}')
        logging.info(f'Text sample: {text_sample}')
        logging.info(f'Topk image indices: {topk_images}')
        logging.info(f'TopK image logits :{logits}')
        logging.info(f'TopK image entropies: {entropies_all}')
        logging.info('--------------------------------------')
        step+=1
    accuracy = calculate_recall_from_results(reranked_image_results, retrieval_type='image')
    return accuracy




def main(dataset_name,clip_type='CLIPB'):
    path = '/root/dws/MCS/Datasets'

    with open(f'{path}/{dataset_name}_{clip_type}.json', 'r') as f:
            data = json.load(f)
            img_embs = torch.tensor(data['img_embs'])
            text_embs = torch.tensor(data['text_embs'])
            print(f'Loaded {dataset_name} embeddings from json file.')

    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
    scores = text_embs @ img_embs.T
    
    texts_image_index = [i // 1 for i in range(img_embs.shape[0]*1)]
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
    
    T2I_accuracy = T2I_fine_retrieval(texts_image_index,scores, k=5)

    print(f'Final I2T accuracy: {T2I_accuracy:.4f}')
    # I2T_accuracy = I2T_fine_retrieval(texts_image_index,scores, k=5)
    # print(f'Final I2T accuracy: {I2T_accuracy:.3f}')

if __name__ == "__main__":

    dataset_name = 'ShareGPT4V'
    main(dataset_name)
    

    
    
    
