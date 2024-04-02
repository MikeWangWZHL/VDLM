import json
from vllm import LLM, SamplingParams
from openai import OpenAI
import requests
from tqdm import tqdm
import shutil

import sys
import os
current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_file_dir, "../../"))
from inference_util import *


SYSTEM_MESSAGES = {
    "default": "You are a helpful assistant.",
    "svg_expert": "You are a helpful assistant specially trained in understanding, interpreting, and responding to questions about SVG (Scalable Vector Graphics) code."
}
DEFAULT_GENERATION_CONFIGS = {
    "temperature": 0.0,
    "max_tokens": 8192,
    "top_p": 1.0,
}

def load_json(file_path):
    return json.load(open(file_path, "r"))

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def append_to_jsonl(path, data):
    with open(path, 'a') as file:
        json_str = json.dumps(data)
        file.write(json_str + '\n')

def post_process_chat_response(raw_chat_response):
    text_outputs = [{
        "content": choice.message.content,
        "role": choice.message.role,
        "function_call": getattr(choice.message, "function_call", None),
        "tool_calls": getattr(choice.message, "tool_calls", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "index": getattr(choice, "index", None),
    } for choice in raw_chat_response.choices]
    # add prompt to log
    ret = {
        "id": raw_chat_response.id,
        "model": raw_chat_response.model,
        "text_outputs": text_outputs
    }
    return ret



def get_messages_pvd_per_image(img_path, svg_conversion_method="raw_svg_individual_paths_w_rec_check", output_dir=None, svg_config=CONFIGS['default'], sam_generator=None, topk_paths=30, resize_image_before_conversion=False, resize_width=512, diff_threshold=5e-4):
    '''
        convert an image into decomposed svg paths, then form individual prompts for each path to pvd model; output_dir is for storing intermediate information
    '''

    if output_dir is not None:
        svg_output_dir=os.path.join(output_dir, "svg")
        os.makedirs(svg_output_dir, exist_ok=True)
        if svg_conversion_method == "sam_regions_path":
            sam_output_dir=os.path.join(output_dir, "sam_regions")
            os.makedirs(sam_output_dir, exist_ok=True)
        else:
            sam_output_dir = None
        # saving original image
        img = Image.open(img_path)
        img.save(f"{output_dir}/input_img.png")

    if resize_image_before_conversion:
        if output_dir is not None:
            tmp_dir = os.path.join(output_dir, "tmp")
        else:
            tmp_dir = "./tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_img_path = os.path.join(tmp_dir, "tmp_img.png")
        img_path = resize_image(img_path, tmp_img_path, new_width=resize_width)

    svg_strs, svg_paths = img_2_svg_strs(method=svg_conversion_method, img_path=img_path, svg_config=svg_config, svg_output_dir=svg_output_dir, sam_output_dir=sam_output_dir, sam_generator=sam_generator, topk_paths=topk_paths, diff_threshold=diff_threshold)

    # rm tmp_img_path
    if resize_image_before_conversion and os.path.exists(tmp_img_path):
        os.remove(tmp_img_path)
    
    # rm intermediate tmp dir for raw_svg_individual_paths_w_rec_check
    if os.path.exists(os.path.join(svg_output_dir, "tmp")):
        shutil.rmtree(os.path.join(svg_output_dir, "tmp"))

    # pvd prompt
    question = "Describe the visual content of the image in a JSON format."

    messages_per_image = []
    for i, svg_str in enumerate(svg_strs):
        
        prompt = get_prompt_general_from_svg_str(svg_str, question)
        system_prompt = SYSTEM_MESSAGES["svg_expert"]
        
        # ===============
        message = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        # ===============
        messages_per_image.append(message)
    return messages_per_image

def get_task_data_for_pvd_inference(input_json, data_root, output_dir, k=-1, svg_config=CONFIGS['default'], topk_paths=30, resize_image_before_conversion=False, diff_threshold=5e-4):
    '''
        input_json: path to the input json file in data_root/datasets/downstream_tasks, containing:
            - image_path: relative image_path under data_root
            - prompt: original downstream task prompt (not used for pvd inference)
            - label: label for the downstream task (not used for pvd inference)
    '''

    data = load_json(input_json)
    messages = []
    infos = []
    if k > 0:
        print(f"only use the first {k} instances")
        data = data[:k]
    output_dirs_per_image = []
    for item in data:
        image_path = os.path.join(data_root, item["image_path"])
        img_id = item["image_path"].split("/")[-1].split(".")[0]
        output_dir_per_image = os.path.join(output_dir, f"{img_id}")
        messages_per_image = get_messages_pvd_per_image(
            image_path, 
            svg_conversion_method="raw_svg_individual_paths_w_rec_check",
            output_dir=output_dir_per_image,
            svg_config=svg_config,
            topk_paths=topk_paths,
            resize_image_before_conversion=resize_image_before_conversion,
            diff_threshold=diff_threshold
        )
        output_dirs_per_image.append(output_dir_per_image)
        messages.append(messages_per_image)
        infos.append(
            {
                "id": item["id"],
                "image_path": image_path, # absolute path
                "pvd_messages": messages_per_image,
                "instance_data": item, # not used in pvd inference
            }
        )
    return messages, infos, output_dirs_per_image

def setup_client(
        openai_api_key = "EMPTY", 
        openai_api_base = "http://localhost:8000/v1"
    ):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def call_chat_api(client, hosted_model_id, messages, generation_configs = DEFAULT_GENERATION_CONFIGS):
    query_obj = {
        "model": hosted_model_id,
        "messages": messages,
        **generation_configs
    }
    chat_response = client.chat.completions.create(**query_obj)
    return chat_response

import argparse
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="image data root for loading image paths", required=True)
    parser.add_argument("--input_json", type=str, help="Path to the downsteam task input json file in llava format", required=True)
    parser.add_argument("--output_dir", type=str, help="Output dir for both intermediate and final results", required=True)
    parser.add_argument("--k", type=int, help="If only use the first k instances", required=False, default=-1)
    parser.add_argument("--visualize", action='store_true', help="if output visualization of the rendered pvd elements", required=False, default=True)
    parser.add_argument("--model_id", type=str, help="hosted model id, can be found by `curl http://localhost:8000/v1/models` ", required=False, default=None)
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key", required=False, default="EMPTY")
    parser.add_argument("--openai_api_base", type=str, help="OpenAI API base url", required=False, default="http://localhost:8000/v1")

    args = parser.parse_args()

    if args.model_id is None:
        print(f"try automatically find the hosted model id...")
        model_list_data = requests.get(f"{args.openai_api_base}/models").json()
        args.model_id = model_list_data["data"][0]["id"]

    print(f"Using model: {args.model_id}")

    client = setup_client(
        openai_api_key=args.openai_api_key, 
        openai_api_base=args.openai_api_base
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # some default configs
    topk_paths = 30
    resize_image_before_conversion = False
    diff_threshold = 5e-4
    svg_config = CONFIGS['default']
    
    # task specific modifications
    input_task_name = os.path.basename(args.input_json).split(".")[0]
    if "geoclidean" in input_task_name:
        diff_threshold = 1e-6
        svg_config["min_area_ratio"] = None
    elif "nlvr" in input_task_name:
        diff_threshold = 1e-6
        svg_config["min_area_ratio"] = None

        
    messages, infos, output_dirs_per_image = get_task_data_for_pvd_inference(args.input_json, args.data_root, args.output_dir, args.k, svg_config=svg_config, topk_paths=30, resize_image_before_conversion=False, diff_threshold=diff_threshold)
    assert len(messages) == len(infos)

    output_response_jsonl = os.path.join(args.output_dir, "response.jsonl")
    instance_already_processed = set()
    if os.path.exists(output_response_jsonl):
        print("output exists, check existing ids...")
        existing_data = load_jsonl(output_response_jsonl)
        for item in existing_data:
            instance_already_processed.add(item["id"])

    print("total eval num instances:", len(messages))
    print("num instances todo:", len(messages) - len(instance_already_processed))

    generation_configs = DEFAULT_GENERATION_CONFIGS
    print(generation_configs)
    
    for i, messages_per_image in tqdm(enumerate(messages)):

        info = infos[i]
        if info["id"] in instance_already_processed:
            print("instance already processed, skip:", info["id"])
            continue
        
        pvd_raw_responses = []
        pvd_elements = []
        for message in messages_per_image:

            chat_response = call_chat_api(client, args.model_id, message, generation_configs)
            procssed_response = post_process_chat_response(chat_response)
            pvd_raw_responses.append(procssed_response)
            pred_pvd_objs = extract_json_object_pvd(procssed_response["text_outputs"][0]["content"])
            if pred_pvd_objs is not None:
                if not isinstance(pred_pvd_objs, list):
                    pred_pvd_objs = [pred_pvd_objs]
                pvd_elements.append(pred_pvd_objs)

        output = {
            "id": info["id"],
            "pvd_raw_responses": pvd_raw_responses,
            "pvd_elements": pvd_elements,
            "instance_data": info,
        }

        # === visualize ===
        if args.visualize:
            output_visualization_dir = os.path.join(output_dirs_per_image[i], f"output_visualizations")
            os.makedirs(output_visualization_dir, exist_ok=True)
            image_path = info["image_path"]
            img_size = Image.open(image_path).size
            
            if len(pvd_elements) > 1:
                img = Image.new("RGBA", img_size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(img)
                for element in pvd_elements:
                    img = visualize_pvd_shape_prediction(element, img_size, img, draw, alpha=1)
                img.save(f"{output_visualization_dir}/pred_all.png")

        if os.path.exists(output_response_jsonl):
            append_to_jsonl(output_response_jsonl, output)
        else:
            with open(output_response_jsonl, 'w') as file:
                json_str = json.dumps(output)
                file.write(json_str + '\n')
    