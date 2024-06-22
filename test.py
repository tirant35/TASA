from transformers import AutoModelForCausalLM, set_seed
import transformers
from peft import PeftModel, PeftConfig
from contants import PROMPT_DICT, 

prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
set_seed(42)
model_path = "/data/yimin/models/base/meta-llama/Meta-Llama-3-8B"
peft_path = "/data/yimin/peft/peft_trainer/output/MedQA_9800"
example = {
        "instruction": "The following are multiple choice questions (with answers) about medicine.\n\nQuestion: A 33-year-old female presents to her primary care physician complaining of heat intolerance and difficulty sleeping over a one month period. She also reports that she has lost 10 pounds despite no changes in her diet or exercise pattern. More recently, she has developed occasional unprovoked chest pain and palpitations. Physical examination reveals a nontender, mildly enlarged thyroid gland. Her patellar reflexes are 3+ bilaterally. Her temperature is 99\u00b0F (37.2\u00b0C), blood pressure is 135/85 mmHg, pulse is 105/min, and respirations are 18/min. Laboratory analysis is notable for decreased TSH. Which of the following pathophysiologic mechanisms contributed to the cardiovascular symptoms seen in this patient?\nA. Increased numbers of a1-adrenergic receptors\nB. Decreased numbers of a1-adrenergic receptors\nC. Decreased numbers of a2-adrenergic receptors\nD. Decreased sensitivity of \u00df2-adrenergic receptors\nE. Increased sensitivity of \u00df1-adrenergic receptors\nanswer:",
        "input": "",
        "output": "E"
}

model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto', torch_dtype='auto', trust_remote_code=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
peft_config = PeftConfig.from_pretrained(peft_path)
model = PeftModel.from_pretrained(model, peft_path)

input_prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
model_input = tokenizer(input_prompt, return_tensors="pt").to("cuda")
output = model.generate(**model_input, max_new_tokens=1)
out_text = tokenizer.decode(output[0], skip_special_tokens=True).removeprefix(input_prompt)
print(out_text)