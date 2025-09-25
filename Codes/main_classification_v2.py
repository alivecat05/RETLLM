import sys
sys.path.append('../..')
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
from Qwen_vl import Qwen_fine_reasoner
import CLIP.clip as clip
import os
import json
from PIL import Image
import logging
device = "cuda:4" if torch.cuda.is_available() else "cpu"
model, processor = clip.load("ViT-L/14@336px", device=device)
print("CLIP model loaded")

labels_idx = '/root/dws/MCS/classification_datasets/caltech101/label_idx.json'
data_path = '/root/dws/MCS/classification_datasets/caltech101/dataset_list.json'
# awt_path = '/root/dws/MM-embed/Datasets/OxfordFlowers/description.json'
# with open(awt_path, 'r', encoding='utf-8') as f:
#     awt_caption = json.load(f)
def process_json(train_path=None, valid_path=None, test_path=None, labelidx=None):
    def load_json(path, name="data"):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading {name} dataset: {e}")
            return None
    
    train_ds = load_json(train_path, "training") if train_path else None
    valid_ds = load_json(valid_path, "validation") if valid_path else None
    test_ds = load_json(test_path, "testing") if test_path else None
    all_labels = load_json(labelidx, "label index") if labelidx else None

    return train_ds, valid_ds, test_ds, all_labels

def make_dataset_list(dataset_list):
    img_paths = []
    labels = []
    label_idx = []
    problematic_items = []  # 用于记录有问题的数据项

    for item in dataset_list:
        if 'label' not in item or 'file_path' not in item:
            print(f"警告: 数据项 {item} 缺少'label'或'file_path'键")
            problematic_items.append(item)
            continue
        
        label_name = item['label']
        file_path = item['file_path']

        # 检查标签是否能匹配到 all_labels
        matched_idx = None
        for idx, label in all_labels.items():
            if label_name == label:
                matched_idx = idx
                break
        
        if matched_idx is None:
            print(f"警告: 标签 '{label_name}' 无法匹配到 all_labels 中的任何条目")
            problematic_items.append(item)
            continue
        
        # 添加到结果列表
        label_idx.append(matched_idx)
        labels.append(label_name)
        img_paths.append(file_path)

    print(f"数据集中的图像数量: {len(img_paths)}")
    print(f"数据集中的标签数量: {len(labels)}")
    print(f"数据集中的标签索引数量: {len(label_idx)}")
    
    if problematic_items:
        print(f"发现 {len(problematic_items)} 个有问题的数据项:")
        for i, item in enumerate(problematic_items, start=1):
            print(f"{i}. {item}")

    return img_paths, labels, label_idx
class Make_a_Dataset(Dataset):
    def __init__(self, images, labels, label_idxs):
        self.images = images
        self.labels = labels
        self.label_idxs = label_idxs
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        label_idx = self.label_idxs[idx]
        if image is None:
            return None
        return image, label, label_idx
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    images = []
    labels = []
    idx_list = []
    for item in batch:
        image_i, label_i, label_idx = item
        images.append(image_i)
        labels.append(label_i)
        idx_list.append(label_idx)
    
    return {'img_inputs': images, 'idx_list': idx_list}




def get_label_embeddings(all_labels, device):
    label_embeddings = []
    for label in tqdm(all_labels.values(),desc='Processing labels'):
        inputs_text = clip.tokenize([f'a photo of {label}'], truncate=True).to(device)
        label_emb = model.encode_text(inputs_text)
        label_emb = torch.nn.functional.normalize(label_emb, p=2, dim=-1)
        label_embeddings.append(label_emb)
    
    return torch.stack(label_embeddings, dim=0).squeeze(1)




def CLIP_Coarse_classifier(image_path: str,text_embs,labels_idx,top_k: int ,com=[2,21,23,23]):

    image = Image.open(image_path)

    inputs_image = processor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        # image_emb = model.encode_image_multi_layers(inputs_image,strategy='nas_search',group = com) 
        image_emb = model.encode_image(inputs_image)
        image_emb = torch.nn.functional.normalize(image_emb, p=2, dim=-1)
        similarity = image_emb @ text_embs.T
        
        top_k_scores, top_k_indices = torch.topk(similarity.squeeze(), k=top_k)
        top_k_indices = top_k_indices.cpu().tolist()
        top_k_scores = top_k_scores.cpu().tolist()
        top_k_label= []
        for i in top_k_indices:
            top_k_label.append(labels_idx[str(i)])
        pred = top_k_indices[0]   
        
    return pred, top_k_label, top_k_scores

def main(datasets):
    logging.basicConfig(filename='/root/dws/MCS/clsf_logs/caltech101.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    print(f'Loaded dataset with {len(datasets)} samples.')
    accuracy_history = []
    step = 0
    all_preds = []
    R_labels = []
    with torch.no_grad():
        text_embs = get_label_embeddings(all_labels, device)
        for batch in tqdm(All_dataset):
            images = batch['img_inputs']
            idx_list = batch['idx_list']
            
            pred, top_k_label, top_k_scores = CLIP_Coarse_classifier(images[0], text_embs, all_labels,top_k=5)

            # first_key = Qwen_fine_reasoner(image_path=images[0],top_k_labels=top_k_label,is_saliency=False,awt_caption=awt_caption)
            first_key = Qwen_fine_reasoner(image_path=images[0],top_k_labels=top_k_label,is_saliency=False)
            # first_key = Qwen_Reranker_API(image=None,top_k_labels=top_k_label,image_path=images[0])
            # first_key = top_k_label[first_key] if isinstance(first_key, int) else top_k_label[0] 
            Rerank_pred =[key for key, value in all_labels.items() if value == first_key]
            if Rerank_pred != []:
                pred = Rerank_pred[0]
            elif Rerank_pred == []:
                # print(f'Reranker failed,{images[0]}')
                pred = pred
            true_labels = [idx for idx in idx_list]
            Predicted = all_labels[str(pred)]
            G_truth = all_labels[str(idx_list[0])]
            if G_truth == Predicted:
                state = 'True'
            else:
                state = 'False'
            all_preds.append(int(pred))
            R_labels.extend(true_labels)
            R_labels = [int(x) for x in R_labels]
            accuracy = accuracy_score(R_labels, all_preds)
            accuracy = round(accuracy, 4)
            logging.info(f'Steps:{step}/{len(datasets)}  | Is Correct:{state}')
            logging.info(f'Acc: {accuracy:.4f}  ')
            logging.info(f'Correct Label: "{G_truth}" | Predicted Label: "{Predicted}"')
            logging.info(f'Top_K_Labels{top_k_label}')
            logging.info(f'image_path: "{images[0]}')
            logging.info('--------------------------------------')
            if (step + 1) % 10 == 0:
                accuracy_history.append(accuracy)
                print(accuracy)
            step += 1
        print(f'Total Accuracy: {accuracy:.4f}')


    
    
    
    
if __name__ == "__main__":

    train_ds, _, _, all_labels = process_json(data_path, labelidx=labels_idx)
    All_images, All_labels, All_label_idx = make_dataset_list(train_ds)


    All_dataset = Make_a_Dataset(All_images, All_labels, All_label_idx)
    All_dataset = DataLoader(All_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    main(All_dataset)
