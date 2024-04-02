import json
import os
from glob import glob
from copy import deepcopy
import random
from tqdm import tqdm
random.seed(42)

SYSTEM_MESSAGES = {
    "default": "You are a helpful assistant.",
    "svg_expert": "You are a helpful assistant specially trained in understanding, interpreting, and responding to questions about SVG (Scalable Vector Graphics) code."
}

#### === utils === ####
def load_jsonl(jsonl_path):
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def save_jsonl(data, jsonl_path):
    with open(jsonl_path, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f)

def round_floats(obj, precision=2):
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    elif isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    else:
        return obj


def from_llava_json_to_openai_jsonl_format(input_json, output_jsonl_path, system_message=None):
    input_data = json.load(open(input_json))
    output_data = []
    print("input len:", len(input_data))
    for item in input_data:
        d = {
            "id": item["id"],
            "conversations": []
        }
        if system_message is not None:
            d["conversations"].append(
                {
                    "role": "system",
                    "content": system_message
                }
            )
        for c in item['conversations']:
            d["conversations"].append(
                {
                    "role": "user" if c["from"]=="human" else "assistant",
                    "content": c["value"]
                }
            )
        output_data.append(d)
    print("output len:", len(output_data))
    save_jsonl(output_data, output_jsonl_path)


#### === train/eval tasks === ####
def construct_instruction_pair(data_root):

    def get_obj_scene(obj):
        obj_type = obj['type']
        if obj_type == "circle":
            ret = {"type": "circle", "center": obj['center'], "radius": obj['radius']}
        elif obj_type == "ellipse":
            ret = {
                "type": "ellipse", 
                "center": obj['center'], 
                "major_axis_length": obj["major_axis"], 
                "minor_axis_length": obj["minor_axis"], 
                "rotation": obj["angle"]
            }
        elif obj_type == "rectangle":
            ret = {
                "type": "rectangle",
                "vertices": obj['rotated_vertices'],
            }
        elif obj_type == "polygon":
            obj_name = "polygon"
            ret = {
                "type": obj_name,
                "vertices": obj['vertices'],
            }
        elif obj_type in ["triangle", "line_segment"]:
            ret = {
                "type": obj_type,
                "vertices": obj['vertices'],
            }
        elif obj_type in ["arc", "pieslice", "chord"]:
            ret = {
                "type": obj_type,
                "bounding_box": obj['bounding_box'],
                "start_angle": obj['start_angle'],
                "end_angle": obj['end_angle'],
            }
        elif obj_type == "grid":
            vertices = obj['vertices']
            edges = obj['edges']
            ret = {
                "type": "grid",
                "size": [obj['num_points_x'], obj['num_points_y']],
                "vertices": vertices,
                "edges": edges
            }
        elif obj_type in ["graph", "path"]:
            if obj_type == "graph":
                type_literal = "line drawing"
            else:
                type_literal = obj_type

            vertices = obj['vertices']
            if obj_type == "path":
                edges = [[vertices[i], vertices[i+1]] for i in range(len(obj['vertices'])-1)]
            else:
                edges = obj['edges']
            ret = {
                "type": type_literal,
                "vertices": vertices,
                "edges": edges
            }

        # add color
        if obj['style'] == "filled":
            ret['color'] = obj['fill'][:3]
            ret['style'] = "filled shape"
        elif obj['style'] == "outline":
            ret['color'] = obj['outline'][:3]
            ret['line_width'] = obj['width']
            if obj_type not in ["grid", "graph", "path", "line_segment", "arc"]:
                ret['style'] = "outlined shape"
        else:
            ret['fill_color'] = obj['fill'][:3]
            ret['outline_color'] = obj['outline'][:3]
            ret['outline_width'] = obj['width']
            ret['style'] = "filled shape with an outline"
        
        ret = round_floats(ret, 0)
        return ret

    def get_svg_path(img_id, single_obj_dir, multi_obj_filled_dir, multi_obj_outline_dir):
        if "single_obj" in img_id:
            svg_path = os.path.join(single_obj_dir, f"svg/{img_id}.svg")
        elif "multi_obj_filled" in img_id:
            svg_path = os.path.join(multi_obj_filled_dir, f"svg/{img_id}.svg")
        elif "multi_obj_outline" in img_id:
            svg_path = os.path.join(multi_obj_outline_dir, f"svg/{img_id}.svg")
        else:
            raise ValueError(f"Invalid img_id: {img_id}")
        return svg_path

    # TODO: modify this part with custom data
    data_dir = f"{data_root}/pretraining_data/pvd_160k"
    output_path = f"{data_root}/pretraining_data/pvd_160k.json"
    single_obj_dir = os.path.join(data_dir, "pvd_160k-single_obj_100k")
    multi_obj_filled_dir = os.path.join(data_dir, "pvd_160k-multi_obj_filled_20k")
    multi_obj_outline_dir = os.path.join(data_dir, "pvd_160k-multi_obj_outline_40k")
    output_openai_jsonl_path = f"{data_root}/pretraining_data/pvd_160k.jsonl"
    # 
    
    ann_single_obj = load_json(os.path.join(single_obj_dir, "annotations.json"))
    ann_multi_obj_filled = load_json(os.path.join(multi_obj_filled_dir, "annotations.json"))
    ann_multi_obj_outline = load_json(os.path.join(multi_obj_outline_dir, "annotations.json"))
    annotations = {}
    annotations.update(ann_single_obj['data'])
    annotations.update(ann_multi_obj_filled['data'])
    annotations.update(ann_multi_obj_outline['data'])
    
    print("len of annotations:", len(annotations))

    svg_scene_data = []
    items = list(annotations.items())
    random.shuffle(items)
    for img_id, ann in tqdm(items):
        scene = [get_obj_scene(obj) for obj in ann['objects']]
        scene_str = json.dumps(scene)
        
        svg_path = get_svg_path(img_id, single_obj_dir, multi_obj_filled_dir, multi_obj_outline_dir)
        instance_id = f"pvd__{img_id}"

        with open(svg_path, "r") as svg_file:
            svg_content = svg_file.read()
        
        d_svg_scene = {
            "id":instance_id,
            "image":"",
            "conversations":[
                {
                    "from": "human",
                    "value": '''Given an image in SVG format as follows:\n```\n{svg_str}```\n{prompt}'''.format(svg_str=svg_content, prompt="Describe the visual content of the image in a JSON format.")
                },
                {
                    "from": "gpt",
                    "value": scene_str
                }
            ],
            "svg": svg_path.replace(data_root, "")[1:]
        }
        svg_scene_data.append(d_svg_scene)

    # save json in llava / vicuna format
    save_json(svg_scene_data, output_path)

    # save jsonl in openai / mistral format
    from_llava_json_to_openai_jsonl_format(output_path, output_openai_jsonl_path, system_message=SYSTEM_MESSAGES["svg_expert"])
    
if __name__ == "__main__":
    # this script processes the raw pvd_160k svgs and annotations to generate the instruction pair data
    # outputs: (1) llava/vicuna format: pvd_160.json; (2) openai/mistral format: pvd_160k.jsonl
    data_root = "../data/datasets"
    construct_instruction_pair(data_root)
    