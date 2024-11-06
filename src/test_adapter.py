from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import PROMPT_DICT
import json
import os
import fire
from peft import PeftModel
from tqdm import tqdm
import re

def run(
    model_path: str = "/data/yimin/models/base/Qwen/Qwen2-7B",
    adapter_model_id: str = "/data/yimin/peft/TASA/models/adapters/legal/mc",
    data_path: str = "/data/yimin/peft/TASA/data/data_adapters/legal/mc/test.json",
    metric: str = "acc",
    domain: str = "fin",
    task: str = "cmcq"
):

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype='auto', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
    model = PeftModel.from_pretrained(model, adapter_model_id)

    with open(data_path, "r") as f:
        data = json.load(f)
    for example in tqdm(data):
        input_prompt = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        inputs = tokenizer(input_prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=256)
        out_text = tokenizer.decode(output[0], skip_special_tokens=True).removeprefix(input_prompt)
        # out_text = tokenizer.decode(output[0], skip_special_tokens=True)
        example["pred"] = out_text
        print(out_text)
    path = adapter_model_id + "/pred.json"
    if not os.path.exists(path):
        os.mknod(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    
    match metric:
        case "acc":
            acc = 0
            for i in data:
                if domain == 'legal' and task == 'ie':
                    # for legal ie
                    match = re.search(r'\[(.*?)\]', i['output'])
                    if match:
                        content = match.group(1).strip("'")
                        if content == i["pred"]:
                            acc += 1
                    else:
                        if i["pred"] == i["output"] or i["perd"].startswith(i["output"]) or i["output"].startswith(i["pred"]):
                            acc += 1
                else:
                    if i["pred"] == i["output"] or i["pred"].startswith(i["output"]) or i["output"].startswith(i["pred"]):
                            acc += 1
            acc_path = adapter_model_id + "/acc.json"
            if not os.path.exists(acc_path):
                os.mknod(acc_path)
            with open(acc_path, "w") as f:
                f.write(str(acc / len(data)))
        case 'bert':
            import bert_score
            import numpy as np
            cands = []
            refs = []
            for i in data:
                cands.append(i["pred"])
                refs.append(i["output"])
            precision, recall, f1 = bert_score.score(cands, refs, model_type="bert-base-chinese", lang="en", verbose=True)
            precision_mean = precision.mean().item()
            recall_mean = recall.mean().item()
            f1_mean = f1.mean().item()
            bert_score_path = adapter_model_id + "/bert_score.json"
            if not os.path.exists(bert_score_path):
                os.mknod(bert_score_path)
            with open(bert_score_path, "w") as f:
                f.write(f"precision: {precision_mean}, recall: {recall_mean}, f1: {f1_mean}")
                    
        
if __name__ == "__main__":
    fire.Fire(run)