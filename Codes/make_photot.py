from datasets import load_dataset
import os
def load_datasets(data_name,k):
    dataset = load_dataset(f'royokong/{data_name}_test', split='test')
    dataset = dataset.rename_column('text', 'caption')
    dataset = dataset.rename_column('image', 'img')
    if data_name == 'coco':
        dataset = dataset.map(lambda x: {'caption': x['caption'][:5]}, num_proc=4)
    dataset = dataset.select(range(k))
    
    return dataset




if __name__ == "__main__":
    dataset = load_datasets('flickr30k',10)
    print(dataset[0])
    
    images = [item['img'] for item in dataset]
    captions = [item['caption'] for item in dataset]
    
    path = '/root/dws/MCS/Codes/Grouphaha'
    os.makedirs(path,exist_ok=True)
    for i,img in enumerate(images):
        img.save(os.path.join(path,f'{i}.png'))
    with open(os.path.join(path,'captions.txt'),'w') as f:
        for caption in captions:
            f.write(caption[0]+'\n')