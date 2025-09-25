import sys
sys.path.append('/root/dws/MCS')
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
import datasets
# from Codes.Qwen_vl import Qwen_fine_reasoner, Qwen_fine_reasoner_Retrieval_scores_entropy_saliency
import pickle
import CLIP.clip as clip
import os
import json
from PIL import Image
import logging
import numpy as np
def init_clip_model():
    global model, processor,device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model, processor = clip.load("ViT-L/14@336px", device=device)
    print("CLIP model loaded")
    return model, processor

def get_pred(qry_t, tgt_t,normalization=True,k=5):

    if normalization:
        qry_t_norm = torch.linalg.norm(qry_t)
        tgt_t_norms = torch.linalg.norm(tgt_t, axis=1)
        scores = torch.dot(tgt_t, qry_t) / (tgt_t_norms * qry_t_norm)
    else:
        scores = tgt_t @ qry_t.T
    k= min(k, scores.shape[0])
    scores = torch.softmax(scores, dim=0)
    topk_scores, topk_indices = torch.topk(scores, k=k)
    pred = torch.argmax(scores)
    return pred,topk_indices

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
    init_clip_model()

    dataset_name  ='TIGER-Lab/MMEB-eval'
    subset  = 'VisDial'  # 'VizWiz', 'ScienceQA', '-CLI'
    split = 'test'
    img_dir = '/root/dws/MCS/Datasets/MMEB'
    
    remake = True
    
    dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
            )
    eval_qry_dataset = EvalDataset(dataset=dataset,img_dir=img_dir,text_field="qry_text",img_path_field="qry_img_path")
    eval_tgt_dataset = EvalDataset(dataset=dataset,img_dir=img_dir,text_field="tgt_text",img_path_field="tgt_img_path")
    eval_qry_loader = DataLoader(eval_qry_dataset,batch_size=1,collate_fn=eval_collator,shuffle=False,drop_last=False,num_workers=0,)
    eval_tgt_loader = DataLoader(eval_tgt_dataset,batch_size=1,collate_fn=eval_collator,shuffle=False,drop_last=False,num_workers=0,)
    save_dir=f'/root/dws/MCS/Datasets/MMEB/embeddings/{subset}'
    os.makedirs(save_dir, exist_ok=True)
    encode_qry_path = os.path.join(save_dir, 'qry_1')
    encode_tgt_path = os.path.join(save_dir, 'tgt_1')
    if remake:
        if os.path.exists(encode_qry_path):
            os.remove(encode_qry_path)
        if os.path.exists(encode_tgt_path):
            os.remove(encode_tgt_path)
    qry_tensor = []
    if os.path.exists(encode_qry_path) is False:
        with torch.no_grad():
            for batch in tqdm(eval_qry_loader, desc="Encode query"):
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output1, output2 = None, None

                    if isinstance(batch["qry_img_path"], Image.Image):
                        inputs_image = processor(batch["qry_img_path"]).unsqueeze(0).to(device)
                        output1 = model.encode_image(inputs_image)
                        # output1 = torch.nn.functional.normalize(output1, p=2, dim=-1)
                    
                    if isinstance(batch["qry_text"], str):
                        inputs_text = clip.tokenize([batch["qry_text"].replace('<|image_1>','')], truncate=True).to(device)
                        output2 = model.encode_text(inputs_text)
                        # output2 = torch.nn.functional.normalize(output2, p=2, dim=-1)

                    if output1 is not None and output2 is not None:
                        alpha = 0.8  # 图像权重
                        beta = 0.2   # 文本权重
                        output = alpha * output1 + beta * output2
                    elif output1 is not None:
                        output = output1
                    else:
                        output = output2
                    
                    output = torch.nn.functional.normalize(output, p=2, dim=-1)
                qry_tensor.append(output.cpu().detach().float())
        qry_tensor = torch.cat(qry_tensor, dim=0)

        with open(encode_qry_path, 'wb') as f:
            pickle.dump((qry_tensor, eval_qry_dataset.paired_data), f)
    
    if os.path.exists(encode_tgt_path) is False:
        tgt_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output1, output2 = None, None
                    if isinstance(batch["tgt_img_path"], Image.Image):
                        inputs_image = processor(batch["tgt_img_path"]).unsqueeze(0).to(device)
                        output1 = model.encode_image(inputs_image)
                        output1 = torch.nn.functional.normalize(output1, p=2, dim=-1)

                    if isinstance(batch["tgt_text"], str):
                        inputs_text = clip.tokenize([batch["tgt_text"].replace('<|image_1>','')], truncate=True).to(device)
                        output2 = model.encode_text(inputs_text)
                        output2 = torch.nn.functional.normalize(output2, p=2, dim=-1)

                    if output1 is not None and output2 is not None:
                        alpha = 0.7  # 图像权重
                        beta = 0.3   # 文本权重
                        output = alpha * output1 + beta * output2
                    elif output1 is not None:
                        output = output1
                    else:
                        output = output2
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
    logging.basicConfig(filename=f'/root/dws/MCS/logs/MMEB_logs/{subset}_8.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    step=0
    acc_history = []
    for row in tqdm(dataset,desc = 'Calculating accuracy...'):
        qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
        tgt_t, all_candidates = [], []
        for tt in zip(row["tgt_text"], row["tgt_img_path"]):
            tgt_t.append(tgt_dict[tt])
            all_candidates.append(tt[0])
        tgt_t = torch.stack(tgt_t, axis=0)  # (num_candidate, dim)
        pred, top_k_indices = get_pred(qry_t, tgt_t, k=5,normalization=False)
        topk_labels = [all_candidates[i] for i in top_k_indices]
        # query = os.path.join(img_dir,row["qry_img_path"]) if row["qry_img_path"] else row["qry_text"]
        query = [os.path.join(img_dir,row["qry_img_path"]),row["qry_text"].replace('<|image_1|>','')]
        # query = row["qry_text"]
        # first_key = Qwen_fine_reasoner(image_path=query[0], top_k_labels=topk_labels, is_saliency=True,query_text=query[1])
        # first_key = Qwen_fine_reasoner(image_path=query[0], top_k_labels=topk_labels, is_saliency=True,query_text=query[1])
        # first_key, predict, entropies_all=Qwen_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL=topk_labels, captions=query, is_saliency=True, I2T=False, T2I=False)
        # if isinstance(first_key, str):
        #     pred = all_candidates.index(first_key) if isinstance(first_key, str) else top_k_indices[0].item()
        # elif isinstance(first_key, int):
        #     pred = top_k_indices[int(first_key)] if first_key is not None else top_k_indices[0]
        if pred == 0:
            n_correct += 1
            state = True
        else:
            state = False
        acc_history.append({
            'is_correct': True if state else False,
        })
        accuracy = calculate_recall_from_results(acc_history)
        # all_pred.append(all_candidates[pred])
        logging.info(f'Steps:{step}/{len(dataset)}  | Is Correct:{state}')
        logging.info(f'Acc: {accuracy:.4f}  ')
        # logging.info(f'Correct Label: "{all_candidates[0]}" | Predicted Label: "{all_candidates[pred]}"')
        # logging.info(f'Top_K_Labels{topk_labels}')
        # logging.info(f'image_path: "{query}')
        logging.info('--------------------------------------')
        step+=1

    print(f"{subset} accuracy: {n_correct/len(dataset)}")