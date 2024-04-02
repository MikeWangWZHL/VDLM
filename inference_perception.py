import json
from vllm import LLM, SamplingParams
from openai import OpenAI
import requests
from inference_util import *
from prompts import DEFAULT_REASONING_PROMPT

SYSTEM_MESSAGES = {
    "default": "You are a helpful assistant.",
    "svg_expert": "You are a helpful assistant specially trained in understanding, interpreting, and responding to questions about SVG (Scalable Vector Graphics) code."
}
DEFAULT_GENERATION_CONFIGS = {
    "temperature": 0.0,
    "max_tokens": 8192,
    "top_p": 1.0,
}
def call_chat_api(client, hosted_model_id, messages, generation_configs = DEFAULT_GENERATION_CONFIGS):
    query_obj = {
        "model": hosted_model_id,
        "messages": messages,
        **generation_configs
    }
    chat_response = client.chat.completions.create(**query_obj)
    return chat_response

def setup_client(
        openai_api_key = "EMPTY", 
        openai_api_base = "http://localhost:8000/v1"
    ):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def get_perception_from_pvd_responses(responses):
    perception_objs = {}
    for key, item in responses.items():
        obj = json.loads(item['response'])
        if not isinstance(obj, list):
            obj = [obj]
        perception_objs[f"object_{len(perception_objs)}"] = obj
    return perception_objs

import argparse
if __name__ == "__main__":
    # === set up input ===
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="demo_examples/image_inputs/lines_segments.png")
    parser.add_argument("--question", type=str, default="What is the total length of the lines?")
    parser.add_argument("--output_root", type=str, default="demo_examples/perception_output")
    args = parser.parse_args()

    svg_conversion_method = "raw_svg_individual_paths_w_rec_check"
    resize_image_before_conversion = False

    # === set up model client ===
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    print(f"try automatically find the hosted model id...")
    model_list_data = requests.get(f"{openai_api_base}/models").json()
    model_id = model_list_data["data"][0]["id"]
    print(f"Using model: {model_id}")

    client = setup_client(openai_api_base=openai_api_base, openai_api_key=openai_api_key)

    generation_configs = DEFAULT_GENERATION_CONFIGS
    print("generation_configs:", generation_configs)

    model_version = model_id.split("/")[-1]
    img_2_svg_configs=CONFIGS["default"]

    img_path = args.img_path
    if "geoclidean" in img_path or "nlvr" in img_path:
        diff_threshold = 5e-6
    else:
        diff_threshold = 5e-4

    img_base_name = os.path.basename(img_path)
    
    output_dir = f"{args.output_root}/{model_version}/{img_base_name}"

    output_perception_dir = f"{output_dir}/output_perception"
    svg_output_dir=f"{output_dir}/svg"
    
    if resize_image_before_conversion:
        tmp_img_path = f"./tmp_img.png"
        img_path = resize_image(img_path, tmp_img_path, new_width=512)

    os.makedirs(output_dir, exist_ok=True)
    # save original image
    img = Image.open(img_path)
    img.save(f"{output_dir}/input_img.png")
    
    svg_strs, svg_paths = img_2_svg_strs(method=svg_conversion_method, img_path=img_path, svg_config=img_2_svg_configs, svg_output_dir=svg_output_dir, topk_paths=30, diff_threshold=diff_threshold)

    # rm tmp_img_path
    if resize_image_before_conversion and os.path.exists(tmp_img_path):
        os.remove(tmp_img_path)

    # === inference ===
    inputs = []
    responses = []
    for i, svg_str in enumerate(svg_strs):
        print(f"svg_str {i}:", svg_str)
        
        prompt = get_prompt_general_from_svg_str(svg_str, "Describe the visual content of the image in a JSON format.")
        system_prompt = SYSTEM_MESSAGES["svg_expert"]
        
        # ===============
        messages = [
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
        print("input messages:", messages)

        query_obj = {
            "model": model_id,
            "messages": messages,
            **generation_configs
        }
        chat_response = client.chat.completions.create(**query_obj)
        print(chat_response)
        print()
        print(chat_response.choices[0].message.content)
        inputs.append(messages)
        responses.append(chat_response.choices[0].message.content)

    # === save perception results ===
    os.makedirs(output_perception_dir, exist_ok=True)
    w, h = Image.open(img_path).size
    resized_img_size_for_vis = (w, h) # use original size
    response_dict = {}
    for i, response in enumerate(responses):
        svg_path = svg_paths[i]
        svg_basename = os.path.basename(svg_path)
        vis_img = visualize_pvd_shape_prediction(response, resized_img_size_for_vis, alpha=1)
        vis_img.save(f"{output_perception_dir}/pred_{svg_basename}".replace(".svg", ".png"))
        response_dict[svg_basename] = {
            "input": inputs[i],
            "response": response
        }

    with open(f"{output_perception_dir}/responses.json", "w") as f:
        json.dump(response_dict, f, indent=4)

    if len(responses) > 1:
        img = Image.new("RGBA", resized_img_size_for_vis, (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        for response in responses:
            img = visualize_pvd_shape_prediction(response, resized_img_size_for_vis, img, draw, alpha=1)
        img.save(f"{output_perception_dir}/pred_all.png")

    # === construct the prompt for reasoning with the aggregated perception results and the task query ===
    perception_result = get_perception_from_pvd_responses(response_dict)
    propmt_for_reasoning = DEFAULT_REASONING_PROMPT.format(perception=perception_result, question=args.question)
    with open(f"{output_dir}/prompt_for_reasoning.txt", "w") as f:
        f.write(propmt_for_reasoning)