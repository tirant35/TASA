from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/data/yimin/models/base/meta-llama/Meta-Llama-3-8B"
adapter_model_id = "/data/yimin/peft/peft_trainer/output/MedQA_9800"
from constants import PROMPT_DICT, TEST_DIC
example = TEST_DIC['medqa']
prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


input_prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
inputs = tokenizer(input_prompt, return_tensors="pt")

model.load_adapter(adapter_model_id, adapter_name="a1")
model.set_adapter("a1")
output = model.generate(**inputs, max_new_tokens=20)
out_text = tokenizer.decode(output[0], skip_special_tokens=True).removeprefix(input_prompt)
print(out_text)