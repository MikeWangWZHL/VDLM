import os
from glob import glob
import json
from prompts import *
import random
import re
from num2words import num2words
random.seed(42)

# read from environment variable
DATA_ROOT = os.environ.get("DATA_ROOT", "./data/datasets/downstream_tasks")
print("DATA_ROOT:", DATA_ROOT)
PERCEPTION_RESULTS_DIR = os.environ.get("PERCEPTION_RESULTS_DIR", "./results/perception")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_task_data(task_name, dataset_name, prompt_version='v1'):
    system_message = None

    def load_perception_result(instance_id, perception_data):
        if perception_data is None:
            return None
        pvd_elements = perception_data[instance_id]['pvd_elements']
        perception_result = {}
        for item in pvd_elements:
            assert isinstance(item, list)
            perception_result[f"object_{len(perception_result)}"] = item
        return perception_result

    task_name_short, input_type = task_name.split("__")

    if "maze" in task_name_short:
        prompt_type = "maze"
    elif "geoclidean" in task_name_short:
        prompt_type = "geoclidean"
    else:
        prompt_type = task_name_short
    
    print("prompt_type:", prompt_type)

    if input_type == "image":
        prompt_template = downstream_task_image_prompts[prompt_type]
        perception_data = None
    elif "pvd" in input_type: # "pvd_image"
        prompt_template = downstream_task_pvd_prompts[prompt_type]
        perception_result_path = f"{PERCEPTION_RESULTS_DIR}/{task_name_short}/response.jsonl"
        print("IMPORTANT INFO: using perception result:", perception_result_path)
        perception_data = load_jsonl(perception_result_path)
        perception_data = {item['id']: item for item in perception_data}
    
    eval_data = json.load(open(f"{DATA_ROOT}/{task_name_short}.json"))
    
    images = []
    info = []
    prompts = []
    
    for item in eval_data:
        if "image" in input_type: # "image", "pvd_image"
            image_path = os.path.join(DATA_ROOT, item["image_path"])
            assert os.path.exists(image_path)
        else:
            image_path = ""

        images.append(image_path)
        info.append({
            "id": item["id"],
            "label": item["label"],
            "image_path": item["image_path"]
        })
        perception_result = load_perception_result(item["id"], perception_data)

        if task_name_short in ['acute-or-obtuse', 'length-comparison']:
            if perception_result is not None:
                prompt = prompt_template.format(perception=perception_result)
            else:
                prompt = prompt_template
        elif "shapeworld" in task_name_short:
            if perception_result is not None:
                prompt = prompt_template.format(perception=perception_result, question=item["prompt"])
            else:
                prompt = prompt_template.format(question=item["prompt"])
        elif "maze" in task_name_short:
            gsize = int(task_name_short.replace("maze_g", ""))
            if perception_result is not None:
                prompt = prompt_template.format(perception=perception_result, n=gsize, m=gsize-1)
            else:
                prompt = prompt_template.format(n=gsize)
        elif "geoclidean" in task_name_short:
            n_shot = int(task_name_short.split("_")[1].replace("-shot", ""))
            if perception_result is not None:
                prompt = prompt_template.format(perception=perception_result, n_shot=n_shot)
            else:
                prompt = prompt_template.format(n_shot=n_shot)
        elif "nlvr" in task_name_short:
            if perception_result is not None:
                prompt = prompt_template.format(perception=perception_result, question=item["prompt"])
            else:
                prompt = prompt_template.format(question=item["prompt"])
        elif "vgbench_qa_svg" in task_name_short:
            option_str = "\n".join(item['options'])
            if perception_result is not None:
                prompt = prompt_template.format(perception=perception_result, question=item["prompt"], options=option_str)
            else:
                prompt = prompt_template.format(question=item["prompt"], options=option_str)

        prompts.append(prompt)

    return {
        "images": images,
        "info": info,
        "prompts": prompts,
        "system_message": system_message
    }

def parse_output(output_message, **kwargs):
    if "</s>" in output_message:
        parsed_output = output_message.split("</s>")[0].strip(".").strip()
        success_flag = True
    else:
        if isinstance(output_message, str):
            parsed_output = output_message.strip(".").strip()
        else:
            parsed_output = output_message
        success_flag = True
    return parsed_output, success_flag

def get_log_from_vipergpt_response_jsonl(response_jsonl_path):
    log = {"responses":[]}
    with open(response_jsonl_path, 'r') as f:
        for line in f:
            raw_response = json.loads(line)
            if "api_response" not in raw_response or raw_response["api_response"] == []:
                log_response = {
                    "messages": None,
                    "task_data": {
                        "instance_info": raw_response["instance_data"]
                    }
                }
            elif raw_response["api_response"]['code_output'] is None:
                log_response = {
                    "messages": None,
                    "task_data": {
                        "instance_info": raw_response["instance_data"]
                    }
                }
            else:
                log_response = {
                    "system": "Only answer with a function starting def execute_command.",
                    "messages": [
                        ["USER", raw_response["api_response"]["prompt"]],
                        ["ASSISTANT", raw_response["api_response"]["code_input"]],
                        ["ASSISTANT", raw_response["api_response"]["code_output"]],
                    ],
                    "task_data":{
                        "instance_info": raw_response["instance_data"]
                    }
                }
            log["responses"].append(log_response)
    print('loaded len:', len(log["responses"]))
    return log

def get_log_from_gpt_chat_response_jsonl(response_jsonl_path):
    log = {"responses":[]}
    finish_not_by_stop = 0
    with open(response_jsonl_path, 'r') as f:
        for line in f:
            raw_response = json.loads(line)
            if "api_response" not in raw_response or raw_response["api_response"] == []:
                log_response = {
                    "messages": None,
                    "task_data": {
                        "instance_info": raw_response["instance_data"]
                    }
                }
            else:
                log_response = {
                    "system": raw_response["api_response"]["system_message"],
                    "messages": [
                        ["USER", raw_response["api_response"]["prompt"]],
                        ["ASSISTANT", raw_response["api_response"]["text_outputs"][0]["content"]]
                    ],
                    "task_data":{
                        "instance_info": raw_response["instance_data"]
                    }
                }
                if raw_response["api_response"]["text_outputs"][0]['finish_reason'] != "stop":
                    finish_not_by_stop += 1
            log["responses"].append(log_response)
    print("finish_not_by_stop:", finish_not_by_stop)
    print('loaded len:', len(log["responses"]))
    return log

def get_log_from_gpt_assistant_code_interpreter_response_jsonl(response_jsonl_path):
    log = {"responses":[]}
    with open(response_jsonl_path, 'r') as f:
        for line in f:
            raw_response = json.loads(line)
            if "api_response" not in raw_response or raw_response["api_response"] == []:
                log_response = {
                    "messages": None,
                    "task_data": {
                        "instance_info": raw_response["instance_data"]
                    }
                }
            else:
                log_response = {
                    "assistant_name": getattr(raw_response, "assistant_name", "Assistant"),
                    "assistant_instruction": getattr(raw_response, "assistant_instruction", "You are a helpful assistant."),
                    "messages": [
                        ["USER", raw_response["step_details"][0]['messages'][-1]['content']],
                        ["ASSISTANT", raw_response["api_response"]["content"]] # take the final message
                    ],
                    "task_data":{
                        "instance_info": raw_response["instance_data"]
                    }
                }
            log["responses"].append(log_response)
    print('loaded len:', len(log["responses"]))
    # print("finish_not_by_stop:", finish_not_by_stop)
    return log

def extract_json_object(pred, json_type="dict", task="default"):
    try:
        if "```json" in pred:
            json_string = re.search(r"```json(.*?)```", pred, re.DOTALL)
            if json_string is None:
                pred = None
            else:
                json_string = json_string.group(1).strip().replace("\n", "")
                # tuple to list
                json_string = json_string.replace("(", "[").replace(")", "]")
                pred = json.loads(json_string)
        else:
            if json_type == "dict":
                matches = re.findall(r"\{.*?\}", pred, re.DOTALL)
            elif json_type == "list":
                matches = re.findall(r"\[.*?\]", pred, re.DOTALL)

            if len(matches) == 0:
                pred = None
            else:
                json_string = matches[-1].replace("\n", "").replace("\\{", "{").replace("\\}", "}").replace('\\\"', '\"')
                
                if task == "maze":
                    json_string = json_string.replace("\\text", "").replace("\\left", "").replace("\\right", "").replace("\\", "")
                    json_string = re.sub(r"{\s*{.*?}\s*}", "", json_string)
                    json_string = re.sub(r"{\s*{.*?}", "", json_string)
                    json_string = re.sub(r"{.*?}\s*}", "", json_string)
                    json_string = re.sub(r"\[.*?:", "", json_string)
                    json_string = re.sub(r"\{.*?\{", "", json_string)
                    json_string = re.sub(r"\}.*?\}", "", json_string)
                    json_string = re.sub(r"\[\s*?{", "[", json_string)
                    json_string = re.sub(r"}\s*?\]", "]", json_string)

                json_string = json_string.replace("(", "[").replace(")", "]")

                if task == "maze":
                    json_string = re.sub(r'\[\s*\[\s*\[', "[[", json_string)
                    json_string = re.sub(r'\]\s*\]\s*\]', "]]", json_string)
                
                pred = json.loads(json_string)
    except:
        # print("ERROR: cannot load json string")
        pred = None
    return pred 

def get_task_result_from_log(task_name, output_dir, acc_computation_rule = "strict", **kwargs):
    if "assistant_code_interpreter" in os.path.basename(output_dir):
        response_jsonl_path = os.path.join(output_dir, "response.jsonl")
        log = get_log_from_gpt_assistant_code_interpreter_response_jsonl(response_jsonl_path)
    elif "viper" in os.path.basename(output_dir):
        response_jsonl_path = os.path.join(output_dir, "response.jsonl")
        log = get_log_from_vipergpt_response_jsonl(response_jsonl_path)
    elif "gpt" in os.path.basename(output_dir) and "viper" not in os.path.basename(output_dir):
        response_jsonl_path = os.path.join(output_dir, "response.jsonl")
        log = get_log_from_gpt_chat_response_jsonl(response_jsonl_path)
    else:
        log = json.load(open(os.path.join(output_dir, "log.json"))) # llava
    
    gts = []
    preds = []
    output_parsing_success_flags = []
    ids = []

    def parse_json_output_baselayer(task_name, raw_pred, json_type = "dict"):
        key = "solution" if "maze" in task_name else "answer"
        extract_json_task_indicator = "maze" if "maze" in task_name else "default"
        try:
            pred = extract_json_object(raw_pred, json_type=json_type, task=extract_json_task_indicator)
            if isinstance(pred, dict):
                pred = pred[key]
                if isinstance(pred, str) and "maze" in task_name:
                    pred = json.loads(pred)
            elif isinstance(pred, list):
                pred = pred
            else:
                pred = list(pred)
            flag = True
        except Exception as e:
            pred = None
            flag = False
        return pred, flag

    # parse the output and get gts
    for item in log['responses']:
        instance_info = item['task_data']['instance_info']
        ## gt
        gt = item['task_data']['instance_info']['label']
        if isinstance(gt, str):
            gt = gt.lower()

        if "nlvr" in task_name:
            gt = "yes" if gt == "true" else "no"

        ## check if output exists
        if item['messages'] is None:
            preds.append(None)
            gts.append(gt)
            output_parsing_success_flags.append(False)
            ids.append(instance_info["id"])
            continue

        ## parse pred
        raw_pred = item['messages'][-1][-1]

        if "viper" in os.path.basename(output_dir):
            if isinstance(raw_pred, dict):
                try:
                    if "maze" in task_name:
                        pred = raw_pred["solution"]
                    else:
                        pred = raw_pred["answer"]
                    flag = True
                except:
                    flag = False
                    pred = json.dumps(raw_pred)
            else:
                flag = True
                pred = raw_pred

        elif "assistant_code_interpreter" in os.path.basename(output_dir) or "gpt" in os.path.basename(output_dir):
            pred, flag = parse_json_output_baselayer(task_name, raw_pred)

        else: # "llava"
            if "acute-or-obtuse" in task_name or "length-comparison" in task_name:
                pred, flag = parse_output(raw_pred)
            else:
                pred, flag = parse_json_output_baselayer(task_name, raw_pred)    
        if pred is None: # try default parsing
            extract_json_task_indicator = "maze" if "maze" in task_name else "default"
            pred = extract_json_object(raw_pred, json_type="list", task=extract_json_task_indicator)
            if pred is None:
                pred, flag = parse_output(raw_pred)
        
        # Heuristic mapping if the parsing is not successful
        if pred is not None:
            if gt.upper() in ["A", "B", "C", "D"]:
                # string matching for incorrectly parsed outputs
                if pred not in ["A", "B", "C", "D"]:
                    if "A" in pred:
                        pred = "A"
                    elif "B" in pred:
                        pred = "B"
                    elif "C" in pred:
                        pred = "C"
                    elif "D" in pred:
                        pred = "D"
                    else:
                        pred = random.choice(["A", "B", "C", "D"])
            
            if isinstance(pred, str):
                pred = pred.lower()

            if gt in ['yes', 'no']:
                # string matching for incorrectly parsed outputs
                if pred not in ['yes', 'no']:
                    if "correct" in pred or "\"yes\"" in pred:
                        pred = "yes"
                    elif "incorrect" in pred or "\"no\"" in pred:
                        pred = "no"
                    elif "not" in pred:
                        pred = "no"
                    else:
                        pred = random.choice(['yes', 'no']) # if not correctly parsed, pick randomly

            if "acute-or-obtuse" in task_name:
                if pred not in ['acute', 'obtuse']: # if not correctly parsed, pick randomly
                    pred = random.choice(['acute', 'obtuse'])

            if isinstance(gt, str) and gt.isdigit():
                digits_2_words = {i:num2words(int(i)) for i in range(10)}
                for d,w in digits_2_words.items(): # greedy matching
                    if w in pred:
                        pred = str(d)
                        break
        
        preds.append(pred)
        gts.append(gt)
        output_parsing_success_flags.append(flag)
        ids.append(instance_info["id"])
    
    # compute accuracy
    correct = 0
    none_output = 0
    for gt, pred in zip(gts, preds):
        if pred is None:
            none_output += 1
            continue
        if acc_computation_rule == "strict":
            if isinstance(gt, str) and isinstance(pred, str):
                if gt.lower() == pred.lower():
                    correct += 1
            else:
                # print(gt, pred)
                if gt == pred:
                    correct += 1
    if len(gts) == 0:
        acc = 0
    else:
        acc = correct / len(gts)
    print("noun_output: ", none_output, "acc: ", acc, "correct: ", correct, "total: ", len(gts), "task_name: ", task_name)

    return gts, preds, output_parsing_success_flags, ids, acc 

#### helper function for visualization ####
def get_logs_pvd_downstream(model_name='gpt4', input_type="image", result_dir="./results/reasoning", **kwargs):
    logs = []
    for task_name in [
        "acute-or-obtuse",
        "length-comparison",
        "shapeworld-spatial-2obj",
        "shapeworld-spatial-multiobj",
        "shapeworld-superlative",
        "nlvr",
        "geoclidean-2shot",
        "maze-solve-2x2",
        "maze_solve-3x3",
        "vgbench_qa_svg_category",
        "vgbench_qa_svg_color",
        "vgbench_qa_svg_usage",
    ]:
        task_dir = os.path.join(result_dir, f"{task_name}__{input_type}__{model_name}")
        if os.path.exists(task_dir):
            logs.append((f"{task_name}__{input_type}__{model_name}", task_dir))
    return logs

import subprocess
import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', type=str, default="gpt4", help="gpt4 | gpt4v | gpt4o | gpt4_assistant_code_interpreter | vipergpt-gpt4 | llava-v1.5-7b | llava-v1.5-13b")
    parser.add_argument('--input_type', type=str, default="pvd", help="image | pvd | image_pvd")
    parser.add_argument('--result_dir', type=str, default="./results/reasoning")
    args = parser.parse_args()

    model_name = args.model_name
    input_type = args.input_type

    logs = get_logs_pvd_downstream(model_name, input_type, result_dir=args.result_dir)

    acc_computation_rule = "strict"
    for task_name, log_dir in logs:

        gts, preds, output_parsing_success_flags, ids, acc = get_task_result_from_log(task_name, log_dir, acc_computation_rule=acc_computation_rule)
        output_path = os.path.join(log_dir, "results.json")
    
        # output results
        results = {
            "gts": gts,
            "preds": preds,
            "ids": ids,
            "output_parsing_success_flags": output_parsing_success_flags,
            "acc": acc
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        with open(os.path.join(log_dir,"visualize.txt"), 'w') as f:
            f.write(f'GT\tPRED\n')
            for gt, pred in zip(gts, preds):
                gt = str(gt)
                pred = str(pred)
                pred_vis = pred.replace("\n"," ")
                f.write(f'{gt}\t{pred_vis}\n')

        print("log_dir:", log_dir)
        print("gt length:", len(gts))
        print("acc:", acc)
        print("===================")
