from transformers import AutoModelForCausalLM, set_seed
import transformers
from peft import PeftModel, PeftConfig
import torch
import json
import fire
import re
from tqdm import tqdm
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

def initialize_counters(data_dict):
    counters = {'total': {'all': 0}, 'domains': {}}

    for domain, tasks in data_dict.items():
        counters['domains'][domain] = {'all': 0}
        for task in tasks.keys():
            counters['domains'][domain][task] = 0

    return counters
    
def run(
    model_path: str = "/data/yimin/models/base/Qwen/Qwen2-7B",
    selector_path: str = "/data/yimin/peft/TASA/selector/selector_14",
    test_data: str = "/data/yimin/peft/TASA/data/test/test_exp2_500.json",
    data_config_path:str = "/data/yimin/peft/TASA/config/data_config_exp_2.json"
):
    if re.search(r'checkpoint-\d+$', selector_path):
        j = json.load(open(f"{os.path.dirname(selector_path)}/all_results.json", 'r'))
    else:
        j = json.load(open(f"{selector_path}/all_results.json", 'r'))
    if 'test_prompt' in j:
        PROMPT = j['test_prompt']
    else:
        PROMPT = j['prompt']
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto', torch_dtype='auto', trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, selector_path)
    with open(test_data, 'r') as f:
        data = json.load(f)
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    counters = initialize_counters(data_config)
    # Iterate over data with a progress bar
    for i in tqdm(data):
        input_prompt = PROMPT.format_map(i)
        model_input = tokenizer(input_prompt, return_tensors="pt").to("cuda")
        output = model.generate(**model_input, max_new_tokens=25)
        out_text = tokenizer.decode(output[0], skip_special_tokens=True).removeprefix(input_prompt)
        str_list = out_text.split("\n")

        # Skip if the output does not contain enough lines
        if len(str_list) < 2:
            continue

        # Extract domain and task from the output
        domain = str_list[0].removeprefix("domain:")
        task = str_list[1].removeprefix("task:")
        
        print(f"domain: {domain}, task: {task}")

        # Update counters if the domain and task match
        if domain == i['domain'] and task == i['task']:
            counters['total']['all'] += 1
            counters['domains'][domain]['all'] += 1
            counters['domains'][domain][task] += 1


    # Print results
    total_count = counters['total']['all']
    data_length = len(data)
    counters['acc'] = total_count / data_length
    print(f"acc: {total_count / data_length}")

    for domain, tasks in counters['domains'].items():
        print(f"{domain}: {tasks['all']}, {', '.join([f'{k}: {v}' for k, v in tasks.items() if k != 'all'])}")
    
    with open(f"{selector_path}/acc.json", 'w') as f:
        json.dump(counters, f, indent=4)


if "__main__" == __name__:
    fire.Fire(run)