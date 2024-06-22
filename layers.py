from transformers import AutoModelForCausalLM
import torch
from pathlib import Path

def separate_embedding_layer(model: AutoModelForCausalLM, output_dir):
    embedding = model.model.embed_tokens
    torch.save(embedding.state_dict(), Path(output_dir).joinpath('embedding.pth'))
    
def load_embedding_layer(filepath: str):
    state_dict = torch.load(filepath)
    embedding = torch.nn.modules.sparse.Embedding(state_dict['weight'].shape[0], state_dict['weight'].shape[1]).to('cuda')
    embedding.load_state_dict(state_dict)
    return embedding

def run(filepath, outputpath):
    model = AutoModelForCausalLM.from_pretrained(filepath, device_map='auto')
    separate_embedding_layer(model, outputpath)
    
if __name__ == "__main__":
    model_id = "/data/yimin/models/base/meta-llama/Meta-Llama-3-8B"
    output_dir = "/data/yimin/peft/TASA/llama3-8B"

    run(model_id, output_dir)