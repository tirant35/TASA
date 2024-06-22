from utils import timer, AdapterRegisterer
from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import PROMPT_DICT, TEST_DIC
import logging

logger = logging.getLogger(__name__)
prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
model_id = "/data/yimin/models/base/meta-llama/Meta-Llama-3-8B"
adapters_config = "/data/yimin/peft/TASA/adapter_pool.json"

logger.info("LOAD model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

logger.info("LOAD adapters...")
reg = AdapterRegisterer(model)
reg.add_config_from_json(adapters_config)
model = reg.register(model)

logger.info("TEST:")

adapter_list = ["imdb", "legal", "medqa"]
for adapter in adapter_list:
    text = prompt_input.format_map(TEST_DIC[adapter])
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    model.set_adapter(f"{adapter}_adapter")
    output = model.generate(**inputs, max_new_tokens=10)
    print(tokenizer.decode(output[0], skip_special_tokens=True).removeprefix(text))
    
