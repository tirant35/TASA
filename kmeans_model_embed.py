import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
from layers import separate_embedding_layer, load_embedding_layer
import logging

def load_model_and_tokenizer(modelpath: str, embeddingpath: str, ):

    model = load_embedding_layer(embeddingpath)
    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
    

def get_embeddings(texts, model_id, embedding_id):
    # 加载tokenizer和模型
    model, tokenizer = load_model_and_tokenizer(model_id, embedding_id)
    

    model.eval()
    
    # 将文本编码为token IDs和attention masks
    inputs = [tokenizer(text, return_tensors="pt", padding=True, truncation=True).to('cuda') for text in texts]
    # inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to('cuda')
    # input_ids = inputs['input_ids']

    print("--------------------------------")
    # 获取模型的输出
    outputs = []
    with torch.no_grad():
        from tqdm import tqdm
        for i in tqdm(inputs):
            outputs.append(model(i['input_ids']).flatten().to('cpu').numpy())
            
    # 找出最长数组的长度  
    max_length = max(len(arr) for arr in outputs)
    print(max_length)
  
    # 使用列表推导式和np.pad来填充较短的数组  
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=0) for arr in outputs]  
  
    # 将填充后的数组转换成一个二维NumPy数组  
    embeddings = np.array(padded_arrays)
  
    print(embeddings.shape)

    # 这里使用gpu报错OOM
    # new_size = outputs.size(1) * outputs.size(2)  
    # x_reshaped = outputs.view(outputs.size(0), -1)
    # embeddings = x_reshaped.cpu().numpy().squeeze()

    return embeddings

def cluster_embeddings_with_faiss(embeddings, n_clusters):
    logging.info("Kmeans...")
    # 转换为FAISS的float32格式
    embeddings_faiss = embeddings.astype(np.float32)
    # embeddings_faiss = embeddings
    # 创建FAISS索引
    d = embeddings_faiss.shape[1]  # 数据维度
    index = faiss.IndexFlatL2(d)  # 使用L2距离

    # 添加数据到索引
    index.add(embeddings_faiss)

    # 进行K-means聚类
    # niter：迭代次数，verbose：log全部输出
    # kmeans = faiss.Kmeans(d, n_clusters, niter=50, verbose=True)
    kmeans = faiss.Kmeans(d, n_clusters)
    kmeans.train(embeddings_faiss)
    # 如下是返回每个点的标签
    # _, labels = kmeans.index.search(embeddings_faiss, 1)
    # 如下是返回对应每个中心的索引
    _, labels = index.search(kmeans.centroids, 1)

    return labels.flatten()



# texts = ["这是一段文本。", "这是另一段文本。", "这两段文本在语义上有所不同。", "这两段文本在语义上相同。"]
model_name = "/data/yimin/models/base/meta-llama/Meta-Llama-3-8B"
embedding_path = "/data/yimin/peft/TASA/llama3-8B/embedding.pth"
n_clusters = 100  # 聚类数为100

import json
with open('/data/yimin/dataset/train/train_only/im_train.json', 'r') as f:
    data = json.load(f)
    
from constants import PROMPT_DICT

prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
texts = [prompt_input.format_map(i) for i in data]

embeddings = get_embeddings(texts=texts, model_id=model_name, embedding_id=embedding_path)
labels = cluster_embeddings_with_faiss(embeddings, n_clusters)

# 输出聚类结果
for i in labels:
    print(f"聚类中心: {texts[i]}")
# for i, text in enumerate(texts):
#     print(f"文本: {texts}, 聚类标签: {labels[i]}")
print(labels)
