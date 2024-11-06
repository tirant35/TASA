import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
from layers import separate_embedding_layer, load_embedding_layer
import logging
import json
from constants import PROMPT_DICT

logger = logging.getLogger('sentence_transformers')

# 将其日志级别设为 ERROR 或更高，以减少输出
logger.setLevel(logging.WARNING)

def load_model_and_tokenizer(modelpath: str, embeddingpath: str, ):

    model = load_embedding_layer(embeddingpath)
    tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
    

def get_embeddings(texts, model_id, embedding_id):
    # 加载tokenizer和模型
    model, tokenizer = load_model_and_tokenizer(model_id, embedding_id)
    

    model.eval()
    
    # 将文本编码为token IDs和attention masks
    # inputs = [tokenizer(text, return_tensors="pt", padding=True, truncation=True).to('cuda') for text in texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to('cuda:1')
    inputs_ids = inputs['input_ids']

    output_averages = []

    with torch.no_grad():
        from tqdm import tqdm
        for i in tqdm(inputs_ids):
            output = model(i)
            output_average = torch.mean(output, dim=0)
            output_averages.append(output_average.cpu().numpy())
    output_averages = np.array(output_averages).squeeze()

    del model
    
    return output_averages

def cluster_embeddings_with_faiss(embeddings, n_clusters, distance):
    logging.info("Kmeans...")
    # 转换为FAISS的float32格式
    embeddings_faiss = embeddings.astype(np.float32)
    # embeddings_faiss = embeddings
    # 创建FAISS索引
    d = embeddings_faiss.shape[1]  # 数据维度
    if distance == "L2":
        logger.warning("Using L2 distance")
        index = faiss.IndexFlatL2(d)
    elif distance == "IP":
        logger.warning("Using IP distance")
        index = faiss.IndexFlatIP(d)
    elif distance == "COS":
        # Normalization plus inner product equals cosine similarity
        norms = np.linalg.norm(embeddings_faiss, axis=1, keepdims=True)
        embeddings_faiss = embeddings_faiss / norms   
        logger.warning("Using cosine distance")
        index = faiss.IndexFlatIP(d)
    else:
        raise ValueError("Invalid distance metric")

    # 添加数据到索引
    index.add(embeddings_faiss)

    # 进行K-means聚类
    # niter：迭代次数，verbose：log全部输出
    # kmeans = faiss.Kmeans(d, n_clusters, niter=50, verbose=True)
    kmeans = faiss.Kmeans(d, n_clusters, verbose=True)
    kmeans.train(embeddings_faiss)
    # 如下是返回每个点的标签
    # _, labels = kmeans.index.search(embeddings_faiss, 1)
    # 如下是返回对应每个中心的索引
    _, labels = index.search(kmeans.centroids, 1)
    
    logger.warning(f"Kmeans finished:{len(labels.flatten())}")

    return labels.flatten()

def selector_data_embedding(
    data_path: str,
    model_path: str,
    embedding_path: str,
    n_clusters: int,
    domain: str,
    task: str,
    distance: str
):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompt_input = PROMPT_DICT["prompt_input"]
    texts = [prompt_input.format_map(i) for i in data]

    embeddings = get_embeddings(texts=texts, model_id=model_path, embedding_id=embedding_path)
    labels = cluster_embeddings_with_faiss(embeddings, n_clusters, distance)
    return [
        {**item, "domain": domain, "task": task}
        for idx, item in enumerate(data)
        if idx in labels
    ]

if "__main__" == __name__:

    # texts = ["这是一段文本。", "这是另一段文本。", "这两段文本在语义上有所不同。", "这两段文本在语义上相同。"]
    model_name = "/data/yimin/models/base/Qwen/Qwen2-7B"
    embedding_path = "/data/yimin/peft/TASA/models/Qwen2-7B/embedding.pth"
    n_clusters = 100  # 聚类数为100

    with open('/data/yimin/peft/TASA/data/data_adapters/fin/cmcq/train.json', 'r') as f:
        data = json.load(f)
        
    from constants import PROMPT_DICT

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    texts = [prompt_input.format_map(i) for i in data]

    embeddings = get_embeddings(texts=texts, model_id=model_name, embedding_id=embedding_path)
    labels = cluster_embeddings_with_faiss(embeddings, n_clusters, 'L2')

    # 输出聚类结果
    for i in labels:
        print(f"聚类标签: {i}")
    label_list = [0] * len(embeddings)
    for index, label in enumerate(label_list):
        if index in labels:
            label_list[index] = 1
    print(sum(label_list))
    # for i in labels:
    #     print(f"聚类中心: {texts[i]}")
    # for i, text in enumerate(texts):
    #     print(f"文本: {texts}, 聚类标签: {labels[i]}")
