import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
import sys
sys.path.insert(0, '/root/dws/MCS/Codes')
from Qwen_vl import Qwen_fine_compare


def load_model(clip_type,device):
    if clip_type == "Long-CLIP":
        sys.path.insert(0, '/root/dws/MCS/Long-CLIP')
        sys.path.insert(0, '/root/dws/MCS/Long-CLIP/model')
        from model import longclip as clip
        clip_model, preprocess = clip.load(
            "/root/dws/MCS/Long-CLIP/checkpoints/longclip-L.pt",  # 使用ViT-L/14作为Long-CLIP模型
            device=device
        )
    elif clip_type == "CLIP":
        sys.path.insert(0, '/root/dws/MCS/CLIP')
        import clip
        clip_model, preprocess = clip.load(
            "ViT-L/14@336px",  # 使用ViT-L/14作为Long-CLIP模型
            device=device
        )
    tokenizer = clip.tokenize
    print(f" {clip_type} model loaded successfully")
    return clip_model, tokenizer,preprocess


@torch.no_grad()
def text_retrieval(pos_text, neg_text, image, model, tokenizer, transform, device):
    pos_text = tokenizer(pos_text).to(device)
    pos_text_embedding = model.encode_text(pos_text)
    neg_text = tokenizer(neg_text).to(device)
    neg_text_embedding = model.encode_text(neg_text)
    image_embedding = model.encode_image(transform(image).unsqueeze(dim=0).to(device))
    
    pos_text_embedding /= pos_text_embedding.norm(dim=-1, keepdim=True)
    neg_text_embedding /= neg_text_embedding.norm(dim=-1, keepdim=True)
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    pos_score = pos_text_embedding @ image_embedding.t()
    neg_score = neg_text_embedding @ image_embedding.t()
    return 1 if pos_score.item() > neg_score.item() else 0

@torch.no_grad()
def text_retrieval_qwen(pos_text, neg_text, image, model, tokenizer, transform, device):
    text = [neg_text,pos_text]
    ans = Qwen_fine_compare(image_path=image,top_k_labels=text,is_saliency=True)
    if ans is None:
        return 0
    ans = text.index(ans)
    return ans
def evaluate(image_root, dataset, model, tokenizer, transform, device):
    metrics = {}
    for c, data_dict in dataset.items():
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
            image_path = os.path.join(image_root, data['filename'])
            image = Image.open(image_path)
            correct = text_retrieval_qwen(data['caption'], data['negative_caption'], image, model, tokenizer, transform, device)
            correct_cnt += correct
        count = len(data_dict)
        metrics[c] = correct_cnt / count
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="CLIP", help="Model architecture to use from OpenCLIP")
    parser.add_argument('--model_cache_dir', default=None, type=str, help="Directory to where downloaded models are cached")
    parser.add_argument('--output', type=str, default='/root/dws/MCS/Datasets/sugarg/outputs', help="Directory to where results are saved")

    parser.add_argument('--coco_image_root', type=str, default='/root/dws/MCS/Datasets/sugarg/val2017')
    parser.add_argument('--data_root', type=str, default='/root/dws/MCS/Datasets/sugarg/sugar-crepe-main/data')
    parser.add_argument('--all', action="store_true", default=False, help="Whether to test all the pretrained models in the paper")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dict = {
        'add_obj'    : f'{args.data_root}/add_obj.json',
        'add_att'    : f'{args.data_root}/add_att.json',
        'replace_obj': f'{args.data_root}/replace_obj.json',
        'replace_att': f'{args.data_root}/replace_att.json',
        'replace_rel': f'{args.data_root}/replace_rel.json',
        'swap_obj'   : f'{args.data_root}/swap_obj.json',
        'swap_att'   : f'{args.data_root}/swap_att.json',
    }
    dataset = {}
    for c, data_path in data_dict.items():
        dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))

    os.makedirs(args.output, exist_ok=True)

    print(f"Evaluating {args.model}")

    model, tokenizer, transform = load_model(args.model, device)

    metrics = evaluate(args.coco_image_root, dataset, model, tokenizer, transform, device)
    print(metrics)
    print(f"Dump results to: {os.path.join(args.output, f'{args.model}.json')}")
    json.dump(metrics, open(os.path.join(args.output, f'{args.model}.json'), 'w'), indent=4)
