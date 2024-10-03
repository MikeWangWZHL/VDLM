import os
import argparse
import torch

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from tasks import get_task_data
import json
from tqdm import tqdm
import copy

from openai_wrapper import OpenAIWrapper
import time


def load_jsonl(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def append_to_jsonl(path, data):
    with open(path, 'a') as file:
        json_str = json.dumps(data)
        file.write(json_str + '\n')

def remove_instances_stop_by_length(path):
    data = load_jsonl(path)
    new_data = []
    rm_count = 0
    for item in data:
        if item["api_response"]["text_outputs"][0]['finish_reason'] == "stop":
            new_data.append(item)
        else:
            print("remove:", item["api_response"]["id"])
            rm_count += 1
    print("rm_count:", rm_count)
    print("new_data:", len(new_data))
    with open(path, 'w') as file:
        for item in new_data:
            json_str = json.dumps(item)
            file.write(json_str + '\n')



def main(args):
    # set up openai api
    subset_size = args.subset_size
    print("subset_size:", subset_size)
    
    if args.model_type == "gpt4":
        model_name =  "gpt-4-turbo-preview"
    elif args.model_type == "gpt4v":
        model_name =  "gpt-4-vision-preview"
    elif args.model_type == "gpt4o":
        model_name =  "gpt-4o-2024-05-13"
    print("model_name:", model_name)

    if args.api_type == "chat_completion": # "chat_completion", "assistant_code_interpreter"
        openai_config = {
            "api_type": "chat_completion", 
            "model": model_name,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "top_p": 1.0
        }
    elif args.api_type == "assistant_code_interpreter":
        openai_config = {
            "api_type": "assistant_code_interpreter", 
            "assistant_name": args.assistant_name,
            "assistant_instruction": args.assistant_instruction,
            "model": model_name,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "top_p": 1.0,
        }
    if args.response_json:
        print("INFO: response format set to json object")
        openai_config["response_format"] = {"type": "json_object"}
    
    openai_model = OpenAIWrapper(openai_config)
    
    SLEEP_RATE = args.sleep_rate
    print("SLEEP_RATE set to:", SLEEP_RATE)

    # TODO: implement multiple images
    if args.run_task is not None:
        outputs = []
        os.makedirs(args.output_dir, exist_ok=True)
        task_data = get_task_data(args.run_task, args.dataset_name, prompt_version=args.version)

        if args.subset_indices_json is not None:
            print("load subset index from json...")
            subset_idx = json.load(open(args.subset_indices_json, 'r'))
        else:
            if subset_size > len(task_data["images"]):
                print("randomly sample subset index...")
                import random
                random.seed(42)
                subset_idx = random.sample(range(len(task_data["images"])), subset_size)
            else:
                subset_idx = list(range(subset_size))
        print("subset_idx:", subset_idx)
        print("subset_idx len", len(subset_idx))
        
        assert len(subset_idx) == subset_size
        
        with open(os.path.join(args.output_dir, "dataset_config.json"), 'w') as file:
            json.dump({
                "task_name":args.run_task,
                "dataset_name":args.dataset_name,
                "prompt_version":args.version,
                "subset_size":subset_size,
                "subset_idx":subset_idx
            }, file)

        output_path = os.path.join(args.output_dir, "response.jsonl")

        instance_already_processed = set()
        if os.path.exists(output_path):
            print("output exists, check existing ids...")
            existing_data = load_jsonl(output_path)
            for item in existing_data:
                if item != {} and item['api_response'] != []:
                    instance_already_processed.add(item["instance_data"]["id"])


        system_message = task_data["system_message"]
        if system_message is None:
                system_message = ""
        else:
            print("system_message:", system_message)
    
        # run inference on subset
        for idx in subset_idx:
            img_p = task_data["images"][idx]
            text = task_data["prompts"][idx]
            instance_info = task_data["info"][idx]

            if instance_info["id"] in instance_already_processed:
                print("instance already processed, skip:", instance_info["id"])
                continue

            print("running instance:", instance_info["id"])
            if img_p is not None and img_p != "" and os.path.exists(img_p):
                images = [img_p]
            else:
                images = []
            if args.additional_prompt_suffix is not None:
                text += " " + args.additional_prompt_suffix

            
            ret, raw_responses = openai_model.run(prompt=text, images=images, system_message=system_message)
            if args.api_type == "assistant_code_interpreter":
                output = {"api_response":ret, "instance_data":instance_info, "step_details":raw_responses}
            else:
                output = {"api_response":ret, "instance_data":instance_info} 
            outputs.append(output)

            if os.path.exists(output_path):
                append_to_jsonl(output_path, output)
            else:
                with open(output_path, 'w') as file:
                    json_str = json.dumps(output)
                    file.write(json_str + '\n')

            time.sleep(SLEEP_RATE)

    else:        
        # run single inference
        if args.image_file is not None and args.image_file != "" and os.path.exists(args.image_file):
            images = [args.image_file]
        else:
            images = []
        text = args.text_prompt
        if args.additional_prompt_suffix is not None:
            text += " " + args.additional_prompt_suffix
        print(images, text)
        ret, raw_responses = openai_model.run(prompt=text, images=images)
        output = {"api_response":ret, "instance_data":{"text":text, "images":images}}
        print(output)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--additional_prompt_suffix", type=str, default=None)
    parser.add_argument("--subset_size", type=int, default=100)
    parser.add_argument("--subset_indices_json", type=str, default=None)
    parser.add_argument("--model_type", type=str, required=True, default="gpt4")
    parser.add_argument("--text-prompt", type=str, required=False, default=None)
    parser.add_argument('--response_json', action='store_true', help='if strictly return JSON object')
    parser.set_defaults(response_json=False)
    parser.add_argument("--version", type=str, required=False, default="v2")
    parser.add_argument("--image-file", type=str, required=False, default=None)
    parser.add_argument("--run-task", type=str, required=False, default=None)
    parser.add_argument("--dataset-name", type=str, required=False, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sleep_rate", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--api_type", type=str, required=False, help="choose from 'chat_completion', 'assistant_code_interpreter' ", default="chat_completion")
    parser.add_argument("--assistant_name", type=str, required=False, default="Assistant")
    parser.add_argument("--assistant_instruction", type=str, required=False, default="You are a helpful assistant.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()
    main(args)