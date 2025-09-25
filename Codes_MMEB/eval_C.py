import sys


sys.path.append('/root/dws/MCS')

from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
import datasets
import pickle
import gc  # 垃圾回收
# from Qwen_vl import Qwen_fine_reasoner

# 使用Long-CLIP和SEARLE
# sys.path.insert(0, '/root/dws/MCS/Long-CLIP')
# sys.path.insert(0, '/root/dws/MCS/Long-CLIP/model')
# from model import longclip as clip
# # sys.path.insert(0, '/root/dws/MCS/CLIP')
# # import clip

import os
import json
from PIL import Image
import logging
import numpy as np

# 设置内存优化
torch.backends.cudnn.benchmark = True  # 优化CUDNN性能
torch.backends.cudnn.deterministic = False  # 关闭确定性以提高速度

# 全局变量存储模型
clip_model = None
searle_model = None
encode_with_pseudo_tokens = None
preprocess = None

def clear_cache():
    """清理GPU和系统缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理GPU缓存
        torch.cuda.synchronize()  # 同步CUDA操作
    gc.collect()  # 强制垃圾回收
    

def init_models(clip_type:str, device="cuda:2"):
    """初始化CLIP和SEARLE模型"""
    global clip_model, searle_model, encode_with_pseudo_tokens, preprocess,tokenizer
    

    clear_cache()
    
    # 加载SEARLE模型
    clip_model_name = "ViT-L/14"
    searle_model, encode_with_pseudo_tokens = torch.hub.load(
        repo_or_dir='/root/dws/MCS/SEARLE', 
        source='local', 
        model='searle',
        backbone=clip_model_name
    )
    searle_model.to(device)
    

    if clip_type == "Long-CLIP":
        sys.path.insert(0, '/root/dws/MCS/Long-CLIP')
        sys.path.insert(0, '/root/dws/MCS/Long-CLIP/model')
        from model import longclip as clip
        clip_model, preprocess = clip.load(
            "/root/dws/MCS/Long-CLIP/checkpoints/longclip-L.pt",  # 使用ViT-L/14作为Long-CLIP模型
            device=device
        )
        tokenizer = clip.tokenize
    elif clip_type == "CLIP":
        sys.path.insert(0, '/root/dws/MCS/CLIP')
        import clip
        clip_model, preprocess = clip.load(
            "ViT-L/14@336px",  # 使用ViT-L/14作为Long-CLIP模型
            device=device
        )
        tokenizer = clip.tokenize
    print(f"模型初始化完成，使用设备: {device}")
    clear_cache()

    return clip_model, searle_model, encode_with_pseudo_tokens, preprocess,tokenizer

def extract_query_features(image_input, text, clip_model, searle_model, encode_with_pseudo_tokens, preprocess, device):

    if image_input is not None and text and text.strip() and image_input != ''  :

        reference_image = preprocess(image_input).unsqueeze(0).to(device)   
        # reference_features = clip_model.encode_image_multi_layers(reference_image).float()
        reference_features = clip_model.encode_image(reference_image).float()
        pseudo_tokens = searle_model(reference_features.to(device))
        
        # 2. 构建查询特征（参考图像 + 文本描述）
        query_prompt = f"$ {text}".replace('\n','').replace('<|image_1|>Represent the given image with the following question:', '').strip().replace('"','').replace('<|image_1|>','')
        tokenized_query = tokenizer([query_prompt], truncate=True).to(device)
        query_feature = encode_with_pseudo_tokens(clip_model, tokenized_query, pseudo_tokens)
        
        
        
        
        return torch.nn.functional.normalize(query_feature, dim=-1)
    
    elif image_input is not None and image_input != '' and text is None:

        reference_image=preprocess(image_input).unsqueeze(0).to(device)
        reference_features = clip_model.encode_image(reference_image).float()


        return torch.nn.functional.normalize(reference_features, dim=-1)

    elif text and text.strip():
        # 纯文本情况
        tokenized_text = tokenizer([text.strip()], truncate=True).to(device)
        text_features = clip_model.encode_text(tokenized_text).float()
        return torch.nn.functional.normalize(text_features, dim=-1)
    
    else:
        # 默认零向量
        feature_dim = 768
        return torch.zeros(1, feature_dim).to(device)

def extract_target_features(image_input=None, text=None, clip_model=None, preprocess=None, device=None):

    if image_input is not None and text and image_input!='':

        reference_image = preprocess(image_input).unsqueeze(0).to(device)   
        reference_features = clip_model.encode_image(reference_image).float()

        pseudo_tokens = searle_model(reference_features.to(device))

        query_prompt = f"$ {text}".replace('\n','').replace('<|image_1|>Represent the given cropped image of the object', '').strip().replace('"','')
        tokenized_query = tokenizer([query_prompt], truncate=True).to(device)
        query_feature = encode_with_pseudo_tokens(clip_model, tokenized_query, pseudo_tokens)
        return torch.nn.functional.normalize(query_feature, dim=-1)
    elif image_input is not None and image_input!='':
        image = preprocess(image_input).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image).float()
        return torch.nn.functional.normalize(image_features, dim=-1)
    
    elif text and text.strip():
        # 如果只有文本，使用文本特征
        tokenized_text = tokenizer([text.strip()], truncate=True).to(device)
        text_features = clip_model.encode_text(tokenized_text).float()
        return torch.nn.functional.normalize(text_features, dim=-1)
    
    else:
        # 默认零向量
        feature_dim = 768
        return torch.zeros(1, feature_dim).to(device)

def get_pred(qry_t,tgt_t, k=5):

    scores =torch.cosine_similarity(qry_t, tgt_t)
    k = min(k, scores.shape[0])
    topk_scores, topk_indices = torch.topk(scores, k=k)
    pred = torch.argmax(scores)
    return pred, topk_indices, topk_scores

class EvalDataset(Dataset):
    def __init__(self, dataset ,img_dir, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.image_dir = img_dir
        self.eval_data = dataset
        self.text_field = text_field
        self.img_path_field = img_path_field
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })
        self.a = 1
    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, image = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if image is None or image == "":
            image = ""
        else:
            image = self._get_image(image)
        return {
            f"{self.text_field}": text,
            f"{self.img_path_field}": image
        }
        
    
    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.image_dir, img_path)
        image = Image.open(full_img_path)
        return image
    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        # 使用列表代替集合来存储配对，并手动去重
        unique_pair = []
        seen_pairs = set() # 使用一个辅助的集合来进行快速的重复项检查

        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    pair = (row[text_field], row[img_path_field])
                    # 只有在之前没见过这个pair时才添加
                    if pair not in seen_pairs:
                        unique_pair.append(pair)
                        seen_pairs.add(pair)
                else:
                    if isinstance(row[img_path_field], list): # 注意：List 应该是 list
                        for img_path in row[img_path_field]:
                            pair = (row[text_field], img_path)
                            if pair not in seen_pairs:
                                unique_pair.append(pair)
                                seen_pairs.add(pair)
                    else:
                        pair = (row[text_field], row[img_path_field])
                        if pair not in seen_pairs:
                            unique_pair.append(pair)
                            seen_pairs.add(pair)
            elif isinstance(row[text_field], list): # 注意：List 应该是 list
                assert isinstance(row[img_path_field], list) and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    pair = (text, img_path)
                    if pair not in seen_pairs:
                        unique_pair.append(pair)
                        seen_pairs.add(pair)

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data
class EvalCollator:
    def __call__(self, batch):
        return batch[0]
def calculate_recall_from_results(results) -> float:

    total_samples = len(results)
    correct_count = 0
    if total_samples == 0:
        return 0.0
    correct_count = sum(1 for res in results if res['is_correct'])
    recall = correct_count / total_samples
    return recall   
eval_collator = EvalCollator()
    
if __name__ == "__main__":

    from Codes.Qwen_vl import *
    device = "cuda:4"
    clip_model, searle_model, encode_with_pseudo_tokens, preprocess, tokenizer = init_models("Long-CLIP",device=device)

    dataset_name = 'TIGER-Lab/MMEB-eval'
    subset = 'MSCOCO'  # 'VizWiz', 'ScienceQA', '-CLI'
    split = 'test'
    img_dir = '/root/dws/MCS/Datasets/MMEB'
    
    remake = True  # 是否重新生成特征
    
    dataset = load_dataset(dataset_name,
        subset,
        split=split,
    )
    eval_qry_dataset = EvalDataset(dataset=dataset, img_dir=img_dir, text_field="qry_text", img_path_field="qry_img_path")
    eval_tgt_dataset = EvalDataset(dataset=dataset, img_dir=img_dir, text_field="tgt_text", img_path_field="tgt_img_path")
    eval_qry_loader = DataLoader(eval_qry_dataset, batch_size=1, collate_fn=eval_collator, shuffle=False, drop_last=False, num_workers=0)
    eval_tgt_loader = DataLoader(eval_tgt_dataset, batch_size=1, collate_fn=eval_collator, shuffle=False, drop_last=False, num_workers=0)
    
    save_dir = f'/root/dws/MCS/Datasets/MMEB/embeddings/{subset}'
    os.makedirs(save_dir, exist_ok=True)
    encode_qry_path = os.path.join(save_dir, 'qry')
    encode_tgt_path = os.path.join(save_dir, 'tgt')
    
    if remake:
        if os.path.exists(encode_qry_path):
            os.remove(encode_qry_path)
        if os.path.exists(encode_tgt_path):
            os.remove(encode_tgt_path)
    
    # 提取查询特征
    qry_tensor = []
    if os.path.exists(encode_qry_path) is False:
        with torch.no_grad():
            for batch in tqdm(eval_qry_loader, desc="Encode query"):
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output = extract_query_features(
                        batch['qry_img_path'], 
                        # None, 
                        batch['qry_text'],
                        clip_model, 
                        searle_model, 
                        encode_with_pseudo_tokens, 
                        preprocess, 
                        device
                    )
                qry_tensor.append(output.cpu().detach().float())
        qry_tensor = torch.cat(qry_tensor, dim=0)

        with open(encode_qry_path, 'wb') as f:
            pickle.dump((qry_tensor, eval_qry_dataset.paired_data), f)
    
    # 提取目标特征
    if os.path.exists(encode_tgt_path) is False:
        tgt_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output = extract_target_features(
                        batch['tgt_img_path'], 
                        None, 
                        # batch['tgt_text'],
                        clip_model, 
                        preprocess, 
                        device
                    )
                tgt_tensor.append(output.cpu().detach().float())
        tgt_tensor = torch.cat(tgt_tensor, dim=0)

        with open(encode_tgt_path, 'wb') as f:
            pickle.dump((tgt_tensor, eval_tgt_dataset.paired_data), f)
    with open(encode_qry_path, 'rb') as f:
        qry_tensor, qry_index = pickle.load(f)
    with open(encode_tgt_path, 'rb') as f:
        tgt_tensor, tgt_index = pickle.load(f)
    qry_dict, tgt_dict = {}, {}
    for qry_t, tt in zip(qry_tensor, qry_index):
        text, img_path = tt["text"], tt["img_path"]
        qry_dict[(text, img_path)] = qry_t
    for tgt_t, tt in zip(tgt_tensor, tgt_index):
        text, img_path = tt["text"], tt["img_path"]
        tgt_dict[(text, img_path)] = tgt_t
        
    n_correct = 0
    all_pred = []
    logging.basicConfig(filename=f'/root/dws/MCS/logs/MMEB_logs/{subset}_C.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    step=0
    acc_history = []
    for row in tqdm(dataset,desc = 'Calculating accuracy...'):
        qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
        tgt_t, all_candidates = [], []
        for tt in zip(row["tgt_text"], row["tgt_img_path"]):
            tgt_t.append(tgt_dict[tt])
            all_candidates.append(tt)
        tgt_t = torch.stack(tgt_t, axis=0)  # (num_candidate, dim)
        # 由于特征已经在提取时进行了归一化，这里使用normalization=True
        pred, top_k_indices, topk_scores = get_pred(qry_t, tgt_t, k=5)
        topk_labels = [all_candidates[i] for i in top_k_indices]
        #topk_labels = [os.path.join(img_dir, label[1]) for label in topk_labels]
        # query = [os.path.join(img_dir,row["qry_img_path"]),row["qry_text"].replace('<|image_1|>','')]
        # query = os.path.join(img_dir,row["qry_img_path"]) if row["qry_img_path"] else row["qry_text"]
        topk_labels = [(t[0].replace('\n','').replace('<|image_1|>Represent the given Wikipedia image with related text information:', '').strip().replace('"',''),os.path.join(img_dir,t[1])) for t in topk_labels]
        # topk_labels = [t[0].replace('\n','').replace('<|image_1|>Represent the given cropped image of the object', '').strip().replace('"','') for t in topk_labels]
        query = (row["qry_text"].replace('\n','').replace('<|image_1|>','').replace('<Represent the given dialogue about an image,', '').strip().replace('"',''), os.path.join(img_dir,row["qry_img_path"])) if row["qry_img_path"] else (row["qry_text"], None)
        # first_key = Qwen_fine_reasoner(image_path=query[1], top_k_labels=topk_labels, is_saliency=True,query_text=query[0])
        #first_key, predict, entropies_all=Qwen_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL=topk_labels, captions=query, is_saliency=True, I2T=False, T2I=False)
        if isinstance(first_key, str):
            pred = all_candidates.index(first_key) if isinstance(first_key, str) else top_k_indices[0].item()
        elif isinstance(first_key, int):
            pred = top_k_indices[int(first_key)] if first_key is not None else top_k_indices[0]
        if pred == 0:
            n_correct += 1
            state = True
        else:
            state = False
        acc_history.append({
            'is_correct': True if state else False,
        })
        accuracy = calculate_recall_from_results(acc_history)
        all_pred.append(all_candidates[pred])
        logging.info(f'Steps:{step}/{len(dataset)}  | Is Correct:{state}')
        logging.info(f'Acc: {accuracy:.4f}  ')
        logging.info(f'Query: "{query[0]}" | Image Path: "{row["qry_img_path"]}"'.replace('\n',''))
        logging.info(f'Correct Label: "{all_candidates[0]}" | Predicted Label: "{all_candidates[pred]}"')
        logging.info(f'Top_K_Labels{topk_labels}')
        # logging.info(f'Top_K_entropy{entropies_all}')
        # logging.info(f'image_path: "{query}')
        logging.info('--------------------------------------')
        step+=1

    print(f"{subset} accuracy: {n_correct/len(dataset)}")