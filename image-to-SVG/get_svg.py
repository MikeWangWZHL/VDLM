import vtracer
from PIL import Image
import os
from vtracer_util import resize_image, post_processing
from configs import CONFIGS, DEFAULT_CONFIG
import json
from tqdm import tqdm
from xml.dom import minidom
from xml.dom.minidom import Document

def get_image_size(dom):
    svg = dom.getElementsByTagName('svg')[0]
    width = svg.getAttribute('width')
    height = svg.getAttribute('height')
    return (int(width), int(height))

def truncate_svg(svg_str, max_str_len=20480, max_num_path = None, output_path=None):

    try:
        # print("truncating svg ...")
        dom = minidom.parseString(svg_str)
        width, height = get_image_size(dom)
        new_doc = Document()
        svg = new_doc.createElement('svg')
        svg.setAttribute('xmlns', "http://www.w3.org/2000/svg")
        svg.setAttribute('version', "1.1")
        svg.setAttribute('width', str(width))
        svg.setAttribute('height', str(height))
        new_doc.appendChild(svg)
        paths = dom.getElementsByTagName('path')
    except Exception as e:
        print("Error in parsing SVG:", e)
        return None

    if max_num_path is not None:
        paths = paths[:max_num_path]

    if len(paths) == 0:
        print("ERROR: No path is found in the SVG; return None!!!")
        return None

    path_count = 0
    truncated_svg_str = None
    for path in paths:
        prev_svg_str = new_doc.toprettyxml(indent="")
        svg.appendChild(path)
        new_svg_str = new_doc.toprettyxml(indent="")

        if len(new_svg_str) > max_str_len:
            truncated_svg_str = prev_svg_str
            break
        else:
            prev_svg_str = new_svg_str
            path_count += 1
    # stoped at the last path
    if truncated_svg_str is None:
        truncated_svg_str = new_doc.toprettyxml(indent="")
    
    if path_count == 0:
        print("ERROR: No path is added during truncation; return empty SVG!!!")

    if output_path:
        with open(output_path, "w") as f:
            f.write(truncated_svg_str)
    return truncated_svg_str

def load_jsonl(jsonl_path):
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def img_to_svg(inp_path, output_path, config = DEFAULT_CONFIG, enforce_overwrite=False):
    if os.path.exists(output_path) and enforce_overwrite is False:
        print(f"output_path: {output_path} exists, skipping ...")
        return
    
    if config["if_resize"] and config["resize_width"] is not None:
        tmp_inp_path = os.path.join(os.path.dirname(inp_path), "tmp_" + os.path.basename(inp_path))
        inp_path = resize_image(inp_path, tmp_inp_path, new_width=config["resize_width"])
    else:
        tmp_inp_path = None

    vtracer.convert_image_to_svg_py(
        inp_path,
        output_path,
        **config["vtracer_config"]
    )

    if config["if_post_process"] and config['vtracer_config']['mode'] == 'polygon':
        post_processing(
            output_path, 
            output_path, 
            min_area_ratio=config["min_area_ratio"], 
            simplify_threshold=config["simplify_threshold"],
            rescale_width=config["rescale_width"]
        )
    
    if tmp_inp_path is not None and os.path.exists(tmp_inp_path):
        os.remove(tmp_inp_path)

    svg_str = open(output_path, "r").read()

    if (config["truncate_len"] is not None and len(svg_str) > config["truncate_len"]) or config["max_num_path"] is not None:
        trunc_len = config["truncate_len"] if config["truncate_len"] is not None else 20480
        max_num_path = config["max_num_path"] if config["max_num_path"] is not None else None
        svg_str = truncate_svg(svg_str, max_str_len=trunc_len, max_num_path=max_num_path, output_path=output_path)
    
    return svg_str

def get_svg_from_instruction_tuning_jsonl(img_data_root, ann_dir, split, config = DEFAULT_CONFIG):
    output_svg_dir = os.path.join(ann_dir, f"svg/{split}")
    os.makedirs(output_svg_dir, exist_ok=True)
    jsonl_path = os.path.join(ann_dir, f"{split}.jsonl")
    data = load_jsonl(jsonl_path)
    for d in tqdm(data):
        inp_path = os.path.join(img_data_root, d['image'])
        if "jpeg" in inp_path:
            output_path = os.path.join(output_svg_dir, os.path.basename(inp_path)[:-5] + ".svg")
        else:
            output_path = os.path.join(output_svg_dir, os.path.basename(inp_path)[:-4] + ".svg")
        img_to_svg(inp_path, output_path, config=config)


# end to end function
def img_to_svg_str(img_path, config, output_path=None, truncate_len=20480):

    if_using_dummy_path = False
    if output_path is None:
        output_path = "tmp.svg" # use a dummy path
        if_using_dummy_path = True
    
    config["truncate_len"] = truncate_len
    svg_str = img_to_svg(img_path, output_path, config, enforce_overwrite=True)
    
    if if_using_dummy_path and os.path.exists(output_path):
        os.remove(output_path)

    return svg_str

from PIL import Image, ImageChops, ImageStat
import cairosvg
import io

def render_svg_to_image(svg_str, output_path=None):
    """
    Renders SVG string to an image using cairosvg and returns a PIL Image.
    """
    png_output = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
    image = Image.open(io.BytesIO(png_output))
    if output_path is not None:
        image.save(output_path)
    return image

def calculate_image_difference(img1, img2, output_path=None):
    """
    Calculates the difference between two images using PIL and returns a normalized difference score.
    """
    # Convert images to the same mode and size for comparison
    img1 = img1.convert('RGBA')
    img2 = img2.convert('RGBA')
    assert img1.size == img2.size

    # Calculate difference and get the sum of absolute values of differences
    diff = ImageChops.difference(img1, img2)
    if output_path is not None:
        diff_rgb = diff.convert('RGB')
        diff_rgb.save(output_path)
    stat = ImageStat.Stat(diff)
    # Normalize the difference score to be between 0 and 1
    # Using sum of squares of differences for each channel
    sum_of_squares = sum(stat.sum2)
    max_diff = 255.0 * 255.0 * 4 * img1.size[0] * img1.size[1]  # Max possible difference
    normalized_diff = sum_of_squares / max_diff

    return normalized_diff

def construct_svg_file_from_single_path(path, width, height, output_path=None):
    new_doc = Document()
    svg = new_doc.createElement('svg')
    svg.setAttribute('xmlns', "http://www.w3.org/2000/svg")
    svg.setAttribute('version', "1.1")
    svg.setAttribute('width', str(width))
    svg.setAttribute('height', str(height))
    new_doc.appendChild(svg)
    svg.appendChild(path)
    path_svg_str = new_doc.toprettyxml(indent="")
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(path_svg_str)
    return path_svg_str

def img_to_individual_svg_paths_w_rec_check(
        inp_path, 
        diff_threshold=5e-4,
        output_dir=None,
        config=CONFIGS['default'], 
        enforce_overwrite=False, 
        topk=None
    ):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_full_img_svg_path = os.path.join(output_dir, os.path.basename(inp_path).replace(".jpg", ".svg").replace(".png", ".svg").replace(".jpeg", ".svg"))
    
    # Convert image to a single SVG with all paths
    svg_str = img_to_svg(inp_path, output_full_img_svg_path, config=config, enforce_overwrite=enforce_overwrite)
    dom = minidom.parseString(svg_str)
    paths = dom.getElementsByTagName('path')
    width, height = get_image_size(dom)
    if topk is not None:
        paths = paths[:topk]

    # make tmp dir for intermediate visualizations
    os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)
    full_svg_rendering_output_path = os.path.join(output_dir, "tmp", "full_img_rendering.png")
    full_rendered_img = render_svg_to_image(svg_str, full_svg_rendering_output_path)  # Assuming this function exists

    # Initialize a document for progressively adding paths
    new_doc = Document()
    svg = new_doc.createElement('svg')
    svg.setAttribute('xmlns', "http://www.w3.org/2000/svg")
    svg.setAttribute('version', "1.1")
    svg.setAttribute('width', str(width))
    svg.setAttribute('height', str(height))
    new_doc.appendChild(svg)

    saved_svg_strs = []
    saved_path_indices = []
    saved_svg_str_files = []
    
    # initialize the previous image as None
    prev_img = None
    
    for i, path in enumerate(paths):
        # Temporarily add path to SVG
        svg.appendChild(path.cloneNode(True))
        temp_svg_str = new_doc.toprettyxml(indent="")

        # Render this temporary SVG and compare with original image
        if output_dir is not None:
            tmp_img_output_path = os.path.join(output_dir, "tmp", f"tmp_img_{i}.png")
        else:
            tmp_img_output_path = None
        temp_img = render_svg_to_image(temp_svg_str, tmp_img_output_path)  # Assuming this function exists
        
        if prev_img is not None:
            if output_dir is not None:
                tmp_per_step_diff_output_path = os.path.join(output_dir, "tmp", f"tmp_per_step_diff_{i}.png")
            else:
                tmp_per_step_diff_output_path = None
            diff = calculate_image_difference(prev_img, temp_img, output_path = tmp_per_step_diff_output_path)  # Assuming this function exists
        else:
            diff = 1 # add the first path

        if diff < diff_threshold:
            # If difference is below threshold, remove this path and skip
            svg.removeChild(svg.lastChild)
            print(f"Path {i} removed due to low difference: {diff}")
        else:
            # Otherwise, keep this path and save the SVG string
            if output_dir is not None:
                output_path_svg_file = os.path.join(output_dir, f"path_{len(saved_svg_strs)}.svg")
                saved_svg_str_files.append(output_path_svg_file) 
            else:
                output_path_svg_file = None
            path_svg_str = construct_svg_file_from_single_path(path, width, height, output_path = output_path_svg_file)
            saved_svg_strs.append(path_svg_str)
            saved_path_indices.append(i)
            
            # If the overall difference with the full image is low enough, break the loop
            if output_dir is not None:
                tmp_overall_diff_output_path = os.path.join(output_dir, "tmp", f"tmp_overall_diff_{i}.png")
            else:
                tmp_overall_diff_output_path = None
            overall_diff = calculate_image_difference(full_rendered_img, temp_img, output_path = tmp_overall_diff_output_path)
            print(f"Path {i} added with per step difference: {diff} | overall difference: {overall_diff}")
            if overall_diff < diff_threshold:
                print(f"Overall dif: {overall_diff} < {diff_threshold}, stopping the addition of new paths.")
                break
            
            # update previous image
            prev_img = temp_img
    
    print(f"Saved {len(saved_svg_strs)} paths out of {len(paths)}")
    print(f"Saved indices {saved_path_indices}")
    return saved_svg_strs, saved_svg_str_files

def img_to_individual_svg_paths(inp_path, output_dir=None, config=CONFIGS['default'], enforce_overwrite=False, topk=None):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_full_img_svg_path = os.path.join(output_dir, os.path.basename(inp_path).replace(".jpg", ".svg").replace(".png", ".svg").replace(".jpeg", ".svg")) 
    
    svg_str = img_to_svg(inp_path, output_full_img_svg_path, config=config, enforce_overwrite=enforce_overwrite)
    dom = minidom.parseString(svg_str)
    paths = dom.getElementsByTagName('path')
    width, height = get_image_size(dom)
    if topk is not None:
        print("saving topk individual paths ...")
        paths = paths[:topk]
    # save each path as an individual svg files
    path_svg_strs = []
    saved_svg_files = []
    for i, path in enumerate(paths):
        new_doc = Document()
        svg = new_doc.createElement('svg')
        svg.setAttribute('xmlns', "http://www.w3.org/2000/svg")
        svg.setAttribute('version', "1.1")
        svg.setAttribute('width', str(width))
        svg.setAttribute('height', str(height))
        new_doc.appendChild(svg)
        svg.appendChild(path)
        path_svg_str = new_doc.toprettyxml(indent="")
        if output_dir is not None:
            saved_svg_file_path = os.path.join(output_dir, f"path_{i}.svg")
            with open(saved_svg_file_path, "w") as f:
                f.write(path_svg_str)
            saved_svg_files.append(saved_svg_file_path)
        path_svg_strs.append(path_svg_str)
    return path_svg_strs, saved_svg_files

def check_svg_path_num(svg_str):
    # check how many paths are there in the svg string
    dom = minidom.parseString(svg_str)
    paths = dom.getElementsByTagName('path')
    return len(paths)