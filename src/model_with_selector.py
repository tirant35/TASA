import transformers
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from constants import PROMPT_DICT
from tabulate import tabulate


PROMPT_INPUT = PROMPT_DICT['prompt_input']

class ModelWithSelector:
    def __init__(self, model_path, selector_path, adapters_pool_dic_path):
        self.model = None
        self.tokenizer = None
        self.selector_path = selector_path
        self.model_path = model_path
        self.adapters_pool_dic = json.load(open(adapters_pool_dic_path, 'r'))
        self.adapters_status_dic = {}
        self.selector_prompt = json.load(open(f"{self.selector_path}/all_results.json", 'r'))['test_prompt']
        for domain, task_list in self.adapters_pool_dic.items():
            for task, adapter in task_list.items():
                self.adapters_status_dic.setdefault(domain, {}).setdefault(task, 0)


    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map='auto', 
            torch_dtype='auto', 
            trust_remote_code=True, 
            # use_flash_attention=False,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
    
    def load_adapters(self):
        for domain, task_list in self.adapters_pool_dic.items():
            for task, adapter in task_list.items():
                self.model.load_adapter(peft_model_id=adapter, adapter_name=f"{domain}_{task}")
                self.adapters_status_dic[domain][task] = 1
                
    
    def load_selector(self):
        self.model.load_adapter(peft_model_id=self.selector_path, adapter_name="selector")
        
    
    def unload_adapter(self, adapter_name):
        self.model.unload_adapter(adapter_name)
        domain = adapter_name.split("_")[0]
        task = adapter_name.split("_")[1]
        self.adapters_status_dic[domain][task] = 0
    
    def generate_adapter(self, data_point, adapter_name, prompt):
        
        test = prompt.format_map(data_point)
        inputs = self.tokenizer(test, return_tensors="pt").to("cuda")
        self.model.set_adapter(adapter_name)
        # outputs = self.model.generate(**inputs, max_new_tokens=100)
        outputs = self.model.generate(**inputs)
        # The following code is temporarily invalid
        # outputs = self.model(**inputs, adapter_names=[adapter_name])
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).removeprefix(test)
    
    
    def generate_selector(self, data_point):
        domain_task = self.generate_adapter(data_point=data_point, adapter_name="selector", prompt=self.selector_prompt)
        print(domain_task)
        domain = domain_task.split("\n")[0].removeprefix("domain:")
        task = domain_task.split("\n")[1].removeprefix("task:")
        return self.generate_adapter(data_point=data_point, adapter_name=f"{domain}_{task}", prompt=PROMPT_INPUT)

    
    def print_adapter_pool(self):
        table_data = []
        for domain, task_list in self.adapters_pool_dic.items():
            for task, adapter in task_list.items():
                table_data.append([domain, task, adapter, self.adapters_status_dic[domain][task]])

        print(tabulate(table_data, headers=["Domain", "Task", "Adapter ID", "Status"], tablefmt="grid"))