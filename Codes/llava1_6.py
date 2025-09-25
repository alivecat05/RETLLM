from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,AutoTokenizer
import torch
from PIL import Image
import requests
import os
import re
device = 'cuda:6'
model_id = '/root/dws/MCS/Models/llama3-llava-next-8b-hf'
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device) 
tokenizer = AutoTokenizer.from_pretrained(model_id)
# prepare image and text prompt, using the appropriate prompt template



def extract_numbers(string):
    # Find all integers and floating-point numbers in the string
    numbers = re.findall(r'\d+\.\d+|\d+', string)
    return numbers
def llavaNext_fine_reasoner(image_path: str, top_k_labels,is_saliency=False, **kwargs):
    label_index = [str(i+1) for i in range(len(top_k_labels))]
    choices_dict = {label_index[i]: top_k_labels[i] for i in range(len(top_k_labels))}

    choices = [f'{label_index[i]}: {top_k_labels[i]}\n' for i in range(len(top_k_labels))]
    choices = ', '.join(choices)
    instruction = (            
        """What is in the picture? """
        # "What type of news is this image related to?"
    )
    prompts = f"""Choose the most relevant scene from the following options : \n{choices}. You should directly output the answer index, for example: 1."""
    template = instruction + '\n'+prompts
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": f"{template}"},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    output_text = processor.decode(output[0], skip_special_tokens=True)
    predict_captions = []
    # print(output_text)
    for output in output_text:
        output = extract_numbers(output.strip().split('\n')[-1])
        if len(output) > 0:
            if output[0] in list(choices_dict.keys()):
                predict_captions.append(choices_dict[output[0]])
    if len(predict_captions) > 0:
        predict_caption = max(set(predict_captions), key=predict_captions.count)
    else:
        predict_caption = None
    return predict_caption
def llavaNext_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL, captions, is_saliency=False, I2T=False, T2I=False):
    predict = []
    entropies_all = []  # 存储所有生成 token 的 entropy
    True_id =tokenizer.convert_tokens_to_ids('True')
    for img in image_PIL:
        conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": f"Does the image match the caption: <{captions}>? Please directly output True or False"},
            {"type": "image"},
            ],
        },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            
        last_token_logits = logits[:, -1, :]
        
        # 数值稳定的softmax计算
        last_token_logits = torch.clamp(last_token_logits, min=-100, max=100)  # 限制logits范围
        probs = torch.softmax(last_token_logits, dim=-1)
        
        # 检查并处理异常值
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"Warning: Found NaN or Inf in probabilities. Using uniform distribution.")
            probs = torch.ones_like(probs) / probs.shape[-1]
        
        # 确保概率值在有效范围内
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        
        # 数值稳定的entropy计算
        log_probs = torch.log(probs + 1e-12)  # 增大epsilon避免log(0)
        entropy = -torch.sum(probs * log_probs, dim=-1).item()
        
        # 检查entropy是否为nan
        if torch.isnan(torch.tensor(entropy)) or torch.isinf(torch.tensor(entropy)):
            # print(f"Warning: Entropy is NaN or Inf. Setting to 0.")
            entropy = 0.0
            
        entropies_all.append(entropy)
        predict_scores = probs[0][True_id].item()  # 获取 True 的概率
        try:
            predict_scores = float(predict_scores)
            predict_scores = int(predict_scores * 100) / 100
        except:
            predict_scores = 0.0

        predict.append(predict_scores)
    try:
        max_score = max(predict)
        
        candidates = [i for i, score in enumerate(predict) if score == max_score]
        
        if len(candidates) == 1:
            best_index = candidates[0]
        else:
            candidate_entropies = [(i, entropies_all[i]) for i in candidates]
            best_index = min(candidate_entropies, key=lambda x: x[1])[0]

        return best_index, predict, entropies_all
    except ValueError:
        return 0, None, None





# 
path = '/root/dws/MCS/Datasets/ShareGPT4V/data/gqa/images'
image_path = [os.path.join(path,f'{i}.jpg') for i in range(1,6)]
image_PIL = []
for image in image_path:
    img = Image.open(image)
    image_PIL.append(img)
i =0 
captions = " A man with glasses is wearing a beer can crocheted hat."
import time
start_time = time.time()
predict_index = llavaNext_fine_reasoner_Retrieval_scores_entropy_saliency(image_PIL,captions, is_saliency=False,I2T = False,T2I = True)
end_time = time.time()
print("Time taken:", end_time - start_time)
print(predict_index)

image_path = '/root/dws/MCS/Datasets/ShareGPT4V/data/gqa/images/1.jpg'
# image_PIL = Image.open(image_path).convert('RGB')
import time
start_time = time.time()
top_k_labels =   ['Man skates along cement wall.', '"A skateboarder rides up a concrete wall, nearly falling off as he tries a trick."', 'A skateboarder is riding his board along the boundary stone of a parking lot.', 'The man does a trick on his skateboard on a concrete ramp.', 'A man does skateboard tricks in a parking lot at night.']
predict_index = llavaNext_fine_reasoner(image_path,top_k_labels,I2T = True,T2I = False)
end_time = time.time()
print("Time taken:", end_time - start_time)
print(predict_index)