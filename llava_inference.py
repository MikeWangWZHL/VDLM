import os
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from tasks import get_task_data
import json
from tqdm import tqdm
import copy

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def _run_model_single_inference(model, tokenizer, image_processor, conv, args):
    """
        conv: conversation object
        args: inference configs
    """
    # load image
    if args.image_file is not None and args.image_file != "":
        image = load_image(args.image_file)
    else:
        image = None
    
    if image is not None:
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    else:
        image_tensor = None

    # add text prompt 
    inp = args.text_prompt

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if image is not None:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    else:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            # streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    
    if args.debug:
        log = {"prompt": prompt, "outputs": outputs}
        print("\n", log, "\n")
    
    return conv.dict()

def main(args):
    # output dir
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, 'log.json')

    # Model
    disable_torch_init()

    # set up model
    if args.model_name is None:
        model_name = get_model_name_from_path(args.model_path)
    else:
        model_name = args.model_name
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name, 
        args.load_8bit, 
        args.load_4bit, 
        device=args.device
    )
    print("model loaded:", model_name)

    # set up conversation
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "vicuna_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # do inference
    all_logs = {
        "configs": copy.deepcopy(args.__dict__),
        "responses":[]
    }
    if args.run_task is not None:
        task_data = get_task_data(args.run_task, args.dataset_name, prompt_version=args.version)
        if args.k > 0:
            print(f"take the first {args.k} instances")
            task_data["images"] = task_data["images"][:args.k]
            task_data["prompts"] = task_data["prompts"][:args.k]
            task_data["info"] = task_data["info"][:args.k]
        
        print("total num instances:", len(task_data["images"]))

        for idx, img_p in tqdm(enumerate(task_data["images"])):
            args.image_file = img_p
            text = task_data["prompts"][idx]
            if args.additional_prompt_suffix is not None:
                text += " " + args.additional_prompt_suffix
            args.text_prompt = text
            instance_info = task_data["info"][idx] 
            conv = conv_templates[args.conv_mode].copy()
            log = _run_model_single_inference(model, tokenizer, image_processor, conv, args)
            log["task_data"] = {
                "instance_info": instance_info,
                "image_path": img_p,
                "idx": idx
            }
            all_logs["responses"].append(log)
            with open(output_path, 'w') as f:
                json.dump(all_logs, f, indent=4)
            # import pdb; pdb.set_trace()
    else:
        # run single inference
        conv = conv_templates[args.conv_mode].copy()
        if args.additional_prompt_suffix is not None:
            args.text_prompt += " " + args.additional_prompt_suffix
        log = _run_model_single_inference(model, tokenizer, image_processor, conv, args)
        log["task_data"] = {"image_path": args.image_file}
        all_logs["responses"].append(log)
        with open(output_path, 'w') as f:
            json.dump(all_logs, f, indent=4)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--text-prompt", type=str, required=False, default=None)
    parser.add_argument("--additional_prompt_suffix", type=str, required=False, default=None)
    parser.add_argument("--version", type=str, required=True, default=None)
    parser.add_argument("--image-file", type=str, required=False, default=None)
    parser.add_argument("--run-task", type=str, required=False, default=None)
    parser.add_argument("--k", type=int, help="if take the first k instances", required=False, default=-1)
    parser.add_argument("--dataset-name", type=str, required=False, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()
    if args.run_task is None:
        assert args.text_prompt is not None, "ERROR: Please specify `--text-prompt`" 
        assert args.image_file is not None, "ERROR: Please specify `--image-file` when --run-task is not specified"
    else:
        assert args.dataset_name is not None, "ERROR: Please specify `--dataset-name` when --run-task is specified"
        print("running task:", args.run_task)
    main(args)
