# Adapters Selector: Cross-domains and Multi-tasks LoRA Modules Integration Usage Method
Parameter-Efficient Fine-Tuning (PEFT) adapts large language models (LLMs) to specific domains by updating only a small portion of the parameters. 
Although fine-tuning on a single task within a specific domain has demonstrated promising results, there remains limited exploration on how to effectively integrate these adapters for optimal performance. 
We propose Adapters Selector (AS): a novel framework for better integrating usage of multiple adapters by training a middleman adapter to select the appropriate adapter for inference.
Our approach utilizes PEFT to train a selector that determines which input content corresponds to which task in which domain, and subsequently selects the homologous adapter.
By the way, The AS has developed the capability to execute cross-domain multi-tasks effectively through the utilization of a compact model in combination with multiple LoRA modules.
## Before Selector Training


### 1. selector training currently supports two methods of data import
（1）Tidy up the data file, construct the following path file and pass it
```
.
├── domain1
│   ├── task1
│   ├── task2
│   └── ...
├── ...
│
```
（2）(recommended) Compile a json data source configuration file, as shown in the following example
``` 
{
    "domain1": {
        "task1": "path1",
        "task2": "path2",
        ...
        },
    ...
}
```
### 2. Separate Embedding Layer
（1）Check the name of the embedding layer of the model. The name of the embedding layer may be different for different model structures and needs to be manually modified. Generally, it is embed_tokens
```
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
print(model.model)
```
（2）We can use layer.py to separate them
## Selector Training
Training with transformers and peft  
`peft_type`:[lora,dora,pissa,relora]  
lora, dora, pissa, rslora and other fine-tuning methods are supported  
`encoder`:[embedding,bert]  
m3e based sentenceEmbedding and model embedding are supported for text embedding  
`distance`:[COS,IP,L2]  
It supports distance measurement methods such as COS, IP and L2  
`n_clusters`:Amount of data to retain for each data (actual retention is less than or equal to this value)  
`use_instruction`:Whether to use instruction for training  
`use_output`:Whether to use output for training

Train with train_selector.py, as shown in the following example
```
python train_selector.py \
    --use_deepspeed false \
    --model_name_or_path /data/yimin/models/base/Qwen/Qwen2-7B \
    --train_data_path /data/yimin/peft/TASA/config/data_config_exp_2.json \
    --output_dir /data/yimin/peft/TASA/selector/exp5/selector_5_13 \
    --peft_type pissa \
    --n_clusters 500 \
    --encoder embedding \
    --distance COS \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --do_train True \
    --use_instruction false \
    --use_output false \
    --bf16 true \
    --fp16 false \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 4e-4 \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 512 \
```
## After Selector Training(model inference with selector)
We will use the ModelWithSelector class from model_with_selector.py to do this reasoning, as shown in the following example  
### 1. Model Loading
```
model = ModelWithSelector(model_path, selector_path, adapters_pool_dic_path)
model.load_model()
model.load_adapters()
model.load_selector()
```
Where adapters_pool_dic_path is the file path of the adapter pool configuration file, which has the following format:
``` 
{
    "domain1": {
        "task1": "adapter1_path",
        "task2": "adapter2_path",
        ...
        },
    ...
}
```
### 2. Model inference
```
model.generate_selector(datapoint)
```
datapoint is the instruction fine-tuning data format, which contains three fields: instruction, input and output

## Selector Update
use update option on selector  
`option`:[add,balance,delete]  
add,balance or delete some domians or tasks  
`option_type`:[new,mix]  
whether to use oringinal training data when adding or balancing  
`reduce_ratio`:[float:0~1]  
the remaining ratio of original training data
```
python update_selector.py \
    --use_deepspeed false \
    --checkpoint_path /data/yimin/peft/TASA/selector/exp6/selector_6_1/checkpoint-230 \
    --embedding_path /data/yimin/peft/TASA/models/Qwen2-7B/embedding.pth \
    --model_name_or_path /data/yimin/models/base/Qwen/Qwen2-7B \
    --update_config_path /data/yimin/peft/TASA/config/task_add_config.json \
    --output_dir /data/yimin/peft/TASA/selector/exp6/selector_6_1_update_3 \
    --option add \
    --option_type mix \
    --reduce_ratio 0.5 \
    --do_train True \
    --use_instruction false \
    --use_output false \
    --bf16 true \
    --fp16 false \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 4e-4 \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 512 \
```
