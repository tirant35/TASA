from transformers import AutoModelForCausalLM
import torch
from pathlib import Path

def separate_embedding_layer(model: AutoModelForCausalLM, output_dir):
    # glm
    # embedding = model.transformer.embedding.word_embeddings
    # internlm
    # embedding = model.model.tok_embeddings
    embedding = model.model.embed_tokens
    torch.save(embedding.state_dict(), Path(output_dir).joinpath('embedding.pth'))
    
def load_embedding_layer(filepath: str):
    state_dict = torch.load(filepath)
    embedding = torch.nn.modules.sparse.Embedding(state_dict['weight'].shape[0], state_dict['weight'].shape[1]).to('cuda:1')
    embedding.load_state_dict(state_dict)
    return embedding

def run(filepath, outputpath):
    model = AutoModelForCausalLM.from_pretrained(filepath, device_map='auto', trust_remote_code=True)
    separate_embedding_layer(model, outputpath)
    
if __name__ == "__main__":
    model_id = "/data/yimin/models/base/internlm/internlm2_5-7b"
    output_dir = "/data/yimin/peft/TASA/internlm2_5-7b"

    run(model_id, output_dir)