import time
from typing import Dict, List
from peft import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import json

class timer():
    
    timetable = []
    current_table = ''
    timetable_dic = {}
    
    def begin(self):
        self.start_time = time.time()
        
    def stop(self, prompt):
        self.end_time = time.time()
        self.time = self.end_time - self.start_time
        self.timetable_dic[self.current_table].append(f"{prompt}:{self.time}")
        
    def add_timetable(self, s):
        self.timetable.append(s)
        self.timetable_dic[s] = []
        self.current_table = s
    
    def set_timetable(self, s):
        self.current_table = s
        
    def print_timetable(self):
        for i in self.timetable:
            print(f'{i}:')
            print('----------------------------------------')
            for out in self.timetable_dic[i]:
                print(out)
            print('----------------------------------------')
            
class AdapterRegisterer():
    
    model_type: str
    path_dic: Dict[str, str] =  {}
    registered_adapter: List[str]
    
    def __init__(self, model: AutoModelForCausalLM):
        self.model_type = model.config.architectures[0]
    
    def add_adapter(self, adapter_name: str, adapter_path: str):
        self.path_dic[adapter_name] = adapter_path
        
    def add_config_from_json(self, pool:str):
        with open(pool, 'r') as f:
            dic = json.load(f)[self.model_type]
        for key, value in dic.items():
            self.add_adapter(key, value)
            
    def register(self, model: AutoModelForCausalLM):
        if self.model_type == model.config.architectures[0]:
            for adapter_name, adapter in self.path_dic.items():
                if adapter_name not in self.registered_adapter:
                    model.load_adapter(peft_model_id=adapter, adapter_name=adapter_name)
                    self.registered_adapter.append(adapter_name)
            return model
        else:
            raise ValueError('The model does not match the registered adapter')
        
def load_data_to_list(data_path:str) -> list:
    file_type = data_path.split('.')[-1]
    try:
        with open(data_path, "r") as f:
            match file_type:
                case "json":
                    return json.load(f)
                case "jsonl":
                    return [json.loads(line) for line in f]
                case _:
                    raise ValueError("Unsupported data types")
    except Exception as e:
        print(f"Exception when loading data: \n {e}")
        
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        