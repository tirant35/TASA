import numpy as np
import faiss
from layers import separate_embedding_layer, load_embedding_layer
from sentence_transformers import SentenceTransformer
import json
from constants import PROMPT_DICT
import fire

def get_embeddings(texts):

    model = SentenceTransformer("/data/gongbu/LLMCraft/m3e-base")
    outputs = []
    from tqdm import tqdm
    for i in tqdm(texts):
        outputs.append(model.encode(i))
    embeddings = np.array(outputs).squeeze()

    return embeddings

def cluster_embeddings_with_faiss(embeddings, n_clusters):

    embeddings_faiss = embeddings.astype(np.float32)
    d = embeddings_faiss.shape[1]  # 数据维度
    index = faiss.IndexFlatL2(d)  # 使用L2距离
    index.add(embeddings_faiss)
    kmeans = faiss.Kmeans(d, n_clusters, verbose=True)
    kmeans.train(embeddings_faiss)
    _, labels = index.search(kmeans.centroids, 1)

    return labels.flatten()


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
    labels = cluster_embeddings_with_faiss(embeddings, n_clusters)

    selected = []
    for i in labels:
        selected.append(data[i])
    with open(output_path, 'w') as f:
        json.dump(selected, f)

if __name__ == '__main__':
    fire.Fire(run)