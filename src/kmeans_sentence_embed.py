import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
import json
from constants import PROMPT_DICT
import fire
import logging

logger = logging.getLogger('sentence_transformers')

# 将其日志级别设为 ERROR 或更高，以减少输出
logger.setLevel(logging.ERROR)

def get_embeddings(texts):

    model = SentenceTransformer("/data/gongbu/LLMCraft/m3e-base")
    outputs = []
    from tqdm import tqdm
    for i in tqdm(texts):
        outputs.append(model.encode(i))
    embeddings = np.array(outputs).squeeze()

    del model
    
    return embeddings

def cluster_embeddings_with_faiss(embeddings, n_clusters, distance):

    embeddings_faiss = embeddings.astype(np.float32)
    d = embeddings_faiss.shape[1]  # 数据维度
    if distance == "L2":
        index = faiss.IndexFlatL2(d)
    elif distance == "IP":
        index = faiss.IndexFlatIP(d)
    elif distance == "COS":
        # Normalization plus inner product equals cosine similarity
        norms = np.linalg.norm(embeddings_faiss, axis=1, keepdims=True)
        embeddings_faiss = embeddings_faiss / norms   
        logger.warning("Using cosine distance")
        index = faiss.IndexFlatIP(d)
    index.add(embeddings_faiss)
    kmeans = faiss.Kmeans(d, n_clusters, verbose=True)
    kmeans.train(embeddings_faiss)
    _, labels = index.search(kmeans.centroids, 1)

    return labels.flatten()

def selector_data(
    data_path: str,
    n_clusters: int,
    domain: str,
    task: str,
    distance: str
):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompt_input = PROMPT_DICT["prompt_input"]
    texts = [prompt_input.format_map(i) for i in data]

    embeddings = get_embeddings(texts=texts)
    labels = cluster_embeddings_with_faiss(embeddings, n_clusters, distance)
    return [
        {**item, "domain": domain, "task": task}
        for idx, item in enumerate(data)
        if idx in labels
    ]

def run(
    data_path: str = "/data/yimin/dataset/train/train_only/im_train.json",
    output_path: str = "/data/yimin/peft/TASA/selected/imdb_selected.json",
    n_clusters: int = 100
    ):
    
    with open(data_path, 'r') as f:
        data = json.load(f)

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    texts = [prompt_input.format_map(i) for i in data]

    embeddings = get_embeddings(texts=texts)
    labels = cluster_embeddings_with_faiss(embeddings, n_clusters, 'COS')

    selected = []
    for i in labels:
        selected.append(data[i])
    with open(output_path, 'w') as f:
        json.dump(selected, f)
        

if __name__ == '__main__':
    model_name = "/data/yimin/models/base/Qwen/Qwen2-7B"
    embedding_path = "/data/yimin/peft/TASA/models/Qwen2-7B/embedding.pth"
    n_clusters = 200  # 聚类数为100

    with open('/data/yimin/peft/TASA/data/data_adapters/fin/cmcq/train.json', 'r') as f:
        data = json.load(f)
        
    from constants import PROMPT_DICT

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    texts = [prompt_input.format_map(i) for i in data]

    embeddings = get_embeddings(texts=texts)
    labels = cluster_embeddings_with_faiss(embeddings, n_clusters, 'L2')

    # 输出聚类结果
    for i in labels:
        print(f"聚类标签: {i}")
    label_list = [0] * len(embeddings)
    for index, label in enumerate(label_list):
        if index in labels:
            label_list[index] = 1
    print(label_list)
    print(sum(label_list))
    # fire.Fire(run)