# from tta_qwen import *
from tta_ZERO import *
# from Qwen_vl import *
import logging
from utils import read_log_groups
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
import torch
from datasets import load_dataset
import os
from typing import List, Dict   
from tqdm import tqdm

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

def load_datasets(data_name,k):
    dataset = load_dataset(f'royokong/{data_name}_test', split='test')
    dataset = dataset.rename_column('text', 'caption')
    dataset = dataset.rename_column('image', 'img')
    if data_name == 'coco':
        dataset = dataset.map(lambda x: {'caption': x['caption'][:5]}, num_proc=4)
    dataset = dataset.select(range(k))

    
    return dataset

def T2I_fine_retrieval():
    logging.basicConfig(filename=f'/root/dws/MCS/tta_logs/coco_TTA_ZERO_ALL_2_1.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    log_file = '/root/dws/MCS/logs/coco_625_t2i.log' 
    reranked_image_results = []
    num_texts = 25000
    
    all_groups = read_log_groups(log_file)
    
    for idx, group_data in enumerate(tqdm(all_groups[7015:10000]),start=7015):
        
        topk_images = group_data.get('Topk image indices', '')
        topk_images = [int(num) for num in topk_images.split(',')]

        text_sample = group_data.get('Text sample', '')
        topk_images = group_data.get('Topk image indices', '')
        topk_images = [int(num) for num in topk_images.split(',')]
        correct_image_index = int(group_data.get('Correct image index', ''))

        path = f'/root/dws/MCS/Datasets/625_coco_images/Group{idx}'
        # image_path = [os.path.join(path,f'topk_image_index_{i}.jpg') for i in range(5)]
        image_path = [os.path.join(path,f'{i}.jpg') for i in range(5)]
        topk_image_samples = []
        for image in image_path:
            img = Image.open(image)
            topk_image_samples.append(img)
        
        reasoned_pred =ZERO_tta(text_sample, topk_image_samples,shuffled_idxes=shuffled_idxes)
        # reasoned_pred, logits, entropies_all = Qwen_fine_reasoner_Retrieval_scores_entropy(image_PIL=topk_image_samples, captions=text_sample)
        try:
            reasoned_pred_index = topk_images[int(reasoned_pred)-1] if reasoned_pred is not None else topk_images[0]
        except:
            reasoned_pred_index = topk_images[0]
        reranked_is_correct = correct_image_index in [reasoned_pred_index]
        reranked_image_results.append({
                    'is_correct': reranked_is_correct,
                })
        accuracy = calculate_recall_from_results(reranked_image_results, retrieval_type='image')

        print(f'{idx}|{num_texts} ,image_retrieval_recall@1: {accuracy:.4f}\n')

        logging.info(f'{idx}|{num_texts} ,image_retrieval_recall@1: {accuracy:.4f}')
        logging.info(f'Is_correct: {reranked_is_correct}')
        logging.info(f'Correct image index: {correct_image_index}, Reasoned image index: {reasoned_pred_index}')
        logging.info(f'Text sample: {text_sample}')
        logging.info(f'Topk image indices: {topk_images}')
        # logging.info(f'Entropies: {entropies_all}')
        # logging.info(f'Logits: {logits}')
        logging.info('--------------------------------------')

    accuracy = calculate_recall_from_results(reranked_image_results, retrieval_type='image')
    return accuracy

def main(dataset_name,k):
    
    T2I_accuracy = T2I_fine_retrieval()

    print(f'Final T2I accuracy: {T2I_accuracy:.3f}')

if __name__ == "__main__":

    dataset_name = 'coco'
    main(dataset_name,k=5000)
    

    
    
    
