import json
import os
import random
import re
from pathlib import Path
from kmeans_sentence_embed import selector_data
from kmeans_model_embed import selector_data_embedding
from utils import load_data_to_list, save_data_to_json
import logging

logger = logging.getLogger(__name__)

def get_domain_task_from_dir(data_dir_path: str):
    """Get the domain list and task list using the total data path from dir path

    Args:
        data_dir_path (str): total data path

    Returns:
        domain_list (str): The string form of domain list
        task_list (str): The string form of task list
    """
    data_path = Path(data_dir_path)
    domain_list = []
    task_set = set()

    for entry in data_path.iterdir():
        if entry.is_dir():
            domain_list.append(entry.name)
            for sub_entry in entry.iterdir():
                if sub_entry.is_dir():
                    task_set.add(sub_entry.name)
    return '[{}]'.format(", ".join(domain_list)), '[{}]'.format(", ".join(task_set))

def get_domain_task_from_config(data_config_path: str):
    """Get the domain list and task list using the total data path from data config

    Args:
        data_dir_path (str): data config path

    Returns:
        domain_list (str): The string form of domain list
        task_list (str): The string form of task list
    """
    domain_list = []
    task_set = set()
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    for domain, task_list in data_config.items():
        domain_list.append(domain)
        for task, _ in task_list.items():
            task_set.add(task)
            
    return '[{}]'.format(", ".join(domain_list)), '[{}]'.format(", ".join(task_set))


def get_domain_task_from_checkpoint(checkpoint_path: str):
    path = Path(checkpoint_path)
    dir_path = path.parent
    data_path = dir_path / "data.json"
    result_path = dir_path / "all_results.json"
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    data_resort = {}
    domain_list = []
    task_list = []
    for item in data:
        domain = item['domain']
        task = item['task']

        if domain not in data_resort:
            data_resort[domain] = {}
            domain_list.append(domain)

        if task not in data_resort[domain]:
            data_resort[domain][task] = []
            task_list.append(task)

        data_resort[domain][task].append(item)
        
    with open(result_path, 'r') as f:
        prompt = json.load(f)['train_prompt']
    
    return data_resort, prompt, domain_list, task_list

def data_balance_option(option_type: str, data_origin: dict, balance_config: dict, **kwargs):

    match option_type:
        case "new":
            return [
                item
                for domain, tasks in balance_config.items()
                for task in tasks
                for item in data_origin[domain][task]
            ]
        case "mix":
            return [
                item
                for domain, tasks in data_origin.items()
                for task in tasks.keys()
                for item in (
                    tasks[task]
                    if domain in balance_config and task in balance_config[domain]
                    else random.sample(
                        tasks[task],
                        int(len(tasks[task]) * kwargs['reduce'])
                        )
                )
            ]
            
def data_add_option(option_type: str, data_origin: dict, add_config: dict, **kwargs):
    new_data = []
    for domain, tasks in add_config.items():
        for task, adapter_data_path in tasks.items():
                    new_data.extend(selector_data_embedding(
                    data_path=adapter_data_path,
                    model_path=kwargs["model_path"],
                    embedding_path=kwargs["embedding_path"],
                    n_clusters=kwargs['n_clusters'],
                    domain=domain,
                    task=task,
                    distance=kwargs['distance']))
    match option_type:
        case "new":
            return new_data
        case "mix":
            for domain, tasks in data_origin.items():
                for task, data in tasks.items():
                    new_data.extend(random.sample(data, int(len(data) * kwargs['reduce'])))
            return new_data
    return

def data_delete_option(option_type: str, data_origin: dict, delete_config: dict, **kwargs):
    new_data = []
    for domain, tasks in data_origin.items():
        for task, data in tasks.items():
            if domain in delete_config and task in delete_config[domain]:
                continue
            else:
                new_data.extend(random.sample(data, int(len(data) * kwargs['reduce'])))
    return new_data

def update_data_from_options(options: str, option_type: str, checkpoint_path: str, update_config: dict, **kwargs):
    data_origin, prompt, _, _ = get_domain_task_from_checkpoint(checkpoint_path)

    match options:
        case 'balance':
            data = data_balance_option(option_type, data_origin, update_config, **kwargs)
            return data, prompt
        case 'add':
            data = data_add_option(option_type, data_origin=data_origin, add_config=update_config, **kwargs)
            pattern_domain = r'Domains:\[(.*?)\]'
            pattern_task = r'Tasks:\[(.*?)\]'
            match_d = re.search(pattern_domain, prompt)
            match_t = re.search(pattern_task, prompt)
            if match_d:
                domain_list = match_d.group(1).split(', ')
            if match_t:
                task_list = match_t.group(1).split(', ')
            for domain, tasks in update_config.items():
                if domain not in domain_list:
                    domain_list.append(domain)
                for task in tasks.keys():
                    if task not in task_list:
                        task_list.append(task)
            prompt = re.sub(pattern_domain, f'Domains:[{", ".join(domain_list)}]', prompt)
            prompt = re.sub(pattern_task, f'Tasks:[{", ".join(task_list)}]', prompt)
            return data, prompt
        case 'delete':
            data = data_delete_option(option_type, data_origin, update_config, **kwargs)
            return data, prompt
            


def construct_selector_data_from_dir(data_dir_path: str, n_clusters: int, distance: str, encoder: str):
    """Constructs the data used to train the selector from the data directory

    Args:
        data_dir_path (str): data dir path
        n_clusters (int): The number of representative data extracted from each dataset

    Returns:
        mix_data (list): The list of data used to train the selector
        
    TODOs:
        1. Add the skill level to the data
        2. multiprocessing
    """
    data_dir = Path(data_dir_path)
    mix_data = []

    for entry in data_dir.rglob('*.json'):  # 使用rglob，更简洁
        if entry.parent != data_dir:  # 确保不在根目录下
            domain = entry.parent.parent.name
            task = entry.parent.name
            skill = None
            try:
                if encoder == 'bert':
                    mix_data.extend(selector_data(entry, n_clusters, domain, task, distance))
                elif encoder == 'embedding':
                    data_embdeeing = selector_data_embedding(
                        data_path=entry,
                        model_path="/data/yimin/models/base/Qwen/Qwen2-7B",
                        embedding_path="/data/yimin/peft/TASA/models/Qwen2-7B/embedding.pth",
                        n_clusters=n_clusters,
                        domain=domain,
                        task=task,
                        distance=distance)
                    mix_data.extend(data_embdeeing)
            except Exception as e:
                print(f"Error processing file {entry}: {e}")
    random.shuffle(mix_data)
    return mix_data


def construct_selector_data_from_config(
    data_config_path: str, 
    n_clusters: int, 
    distance: str, 
    encoder: str,
    **kwargs):
    """Constructs the data set used to train the selector from the data configuration file
        data_config:{
            "domain1": {
                "task1": "path1",
                "task2": "path2"
                ...
            }
            ...
        }
    Args:
        data_config_path (str): Data configuration file path
        n_clusters (int): The number of representative data extracted from each dataset

    Returns:
        mix_data (list): The list of data used to train the selector
        
    TODOs:
        1. Add the skill level to the data
        2. multiprocessing
    """
    mix_data = []
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    for domain, task_list in data_config.items():
        for task, data in task_list.items():
            if encoder == 'bert':
                mix_data.extend(selector_data(data, n_clusters, domain, task, distance))
            elif encoder == 'embedding':
                data_embdeeing = selector_data_embedding(
                    data_path=data,
                    model_path=kwargs["model_path"],
                    embedding_path=kwargs["embedding_path"],
                    n_clusters=n_clusters,
                    domain=domain,
                    task=task,
                    distance=distance)
                mix_data.extend(data_embdeeing)
    random.shuffle(mix_data)
    
    return mix_data


