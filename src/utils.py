import time
from typing import Dict, List
from peft import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import json
import os

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
        

def get_domain_task_skill(data) -> dict:
    domain_list = set()
    task_list = set()
    for i in data:
        domain_list.add(i["domain"])
        task_list.add(i["task"])
        domain_list_str = f'[{", ".join(domain_list)}]'
        task_list_str = f'[{", ".join(task_list)}]'
        skill_list_str = '[]'
    return {'domain': domain_list_str, 'task': task_list_str, 'skill': skill_list_str}

def save_data_to_json(data:list, path:str):
    if path.endswith('.json'):
        if not os.path.exists(path):
            os.mknod(path)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4, default=str)
        except Exception as e:
            print(f"{e}")
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            with open(os.path.join(path, 'data.json'), "w") as f:
                json.dump(data, f, indent=4, default=str)
        except Exception as e:
            print(f"{e}")