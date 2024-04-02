
from PIL import Image, ImageDraw
import os
import json
import sys
import os

# === import image-to-SVG ===
SVG_PROCESSOR_DIR= os.path.join(os.path.dirname(os.path.abspath(__file__)), "image-to-SVG")
sys.path.append(SVG_PROCESSOR_DIR)
from get_svg import (
    img_to_svg_str, 
    img_to_individual_svg_paths, 
    img_to_individual_svg_paths_w_rec_check
)
from configs import CONFIGS


def extract_json_object_pvd(pred):
    try:
        pred = json.loads(pred)
    except:
        print("ERROR: cannot load json string:", pred)
        pred = None
    return pred

def visualize_pvd_shape_prediction(annotations, img_size, img = None, draw = None, alpha=1.0):
    if isinstance(annotations, str):
        annotations = extract_json_object_pvd(annotations)
    
    if annotations is None:
        return None
    
    if not isinstance(annotations, list):
        annotations = [annotations]
    
    if img is None:
        img = Image.new("RGBA", img_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

    for annotation in annotations:
        if "style" not in annotation:
            fill = outline = tuple(annotation["color"] + [int(255 * alpha)])
            width = annotation['line_width']
        else:
            style = annotation["style"]
        
            if style == "filled shape":
                fill = tuple(annotation["color"] + [int(255 * alpha)])
                outline = None
                width = 1
            elif style == "outlined shape":
                outline = tuple(annotation["color"] + [int(255 * alpha)])
                fill = None
                width = annotation['line_width']
            elif style == "filled shape with an outline":
                fill = tuple(annotation["fill_color"] + [int(255 * alpha)])
                outline = tuple(annotation["outline_color"] + [int(255 * alpha)])
                width = annotation['outline_width']
            else:
                raise ValueError(f"Unknown style: {style}")

        w, h = img_size
        line_width_mult = max(w / 512, h / 512, 1)
        width = min(int(width * line_width_mult), 10)
        
        shape_type = annotation["type"]
        if shape_type == "circle":
            center = annotation["center"]
            radius = annotation["radius"]
            bounding_box = [center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius]
            draw.ellipse(bounding_box, fill=fill, outline=outline, width=width)
        
        elif shape_type == "ellipse":
            center = annotation["center"]
            major_axis = annotation["major_axis_length"]
            minor_axis = annotation["minor_axis_length"]
            rotation = annotation["rotation"]

            # Create an ellipse on a separate transparent image
            bounding_box = [center[0]-major_axis, center[1]-minor_axis, center[0]+major_axis, center[1]+minor_axis]
            ellipse_img = Image.new("RGBA", img_size, (0, 0, 0, 0))
            ellipse_draw = ImageDraw.Draw(ellipse_img)
            ellipse_draw.ellipse(bounding_box, fill=fill, outline=outline, width=width)

            # Rotate the ellipse image
            ellipse_img = ellipse_img.rotate(rotation, center=center)

            # Paste the rotated ellipse back onto the main image canvas
            img.paste(ellipse_img, (0, 0), ellipse_img)

        elif shape_type in ["triangle", "polygon", "rectangle", "quadrilateral", "pentagon", "hexagon"]:
            vertices = annotation["vertices"]
            vertices = [tuple(vertex) for vertex in vertices]
            draw.polygon(vertices, fill=fill, outline=outline, width=width)
        
        elif shape_type == "line_segment":
            vertices = annotation["vertices"]
            vertices = [tuple(vertex) for vertex in vertices]
            draw.line(vertices, fill=fill, width=width)
        
        elif shape_type in ["arc", "pieslice", "chord"]:
            bounding_box = annotation["bounding_box"]
            start_angle = annotation["start_angle"]
            end_angle = annotation["end_angle"]
            if shape_type == "arc":
                draw.arc(bounding_box, start_angle, end_angle, fill=fill, width=width)
            elif shape_type == "pieslice":
                draw.pieslice(bounding_box, start_angle, end_angle, fill=fill, outline=outline, width=width)
            elif shape_type == "chord":
                draw.chord(bounding_box, start_angle, end_angle, fill=fill, outline=outline, width=width)

        elif shape_type in ["lines", "line drawing", "grid", "path"]:
            edges = annotation["edges"]
            for edge in edges:
                p1, p2 = edge
                draw.line([tuple(p1), tuple(p2)], fill=fill, width=width)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
    
    return img


def get_prompt_general_from_svg_path(svg_path, prompt):
    prompt_template = '''Given an image in SVG format as follows:\n```\n{svg_str}```\n{prompt}'''
    svg_str = open(svg_path, "r").read()
    prompt = prompt_template.format(svg_str=svg_str, prompt=prompt)
    return prompt
    
def get_prompt_general_from_svg_str(svg_str, prompt):
    prompt_template = '''Given an image in SVG format as follows:\n```\n{svg_str}```\n{prompt}'''
    prompt = prompt_template.format(svg_str=svg_str, prompt=prompt)
    return prompt


def img_2_svg_strs(method = "raw_svg", img_path = None, svg_config = CONFIGS['default'], svg_output_dir = None, sam_output_dir = None, truncate_len = 20480, sam_generator = None, sam_region_seg_configs = None, topk_paths = 30, diff_threshold=5e-4):
    if method == "raw_svg":
        output_path = os.path.join(svg_output_dir, os.path.basename(img_path).replace(".png", ".svg").replace(".jpg", ".svg").replace(".jpeg", ".svg"))
        svg_str = img_to_svg_str(img_path, config=svg_config, output_path=output_path, truncate_len=truncate_len)
        svg_strs = [svg_str]
        svg_paths = [output_path]
    elif method == "raw_svg_individual_paths":
        svg_strs, svg_paths = img_to_individual_svg_paths(img_path, output_dir=svg_output_dir, config=svg_config, enforce_overwrite=True, topk=topk_paths)
    elif method == "raw_svg_individual_paths_w_rec_check":
        svg_strs, svg_paths = img_to_individual_svg_paths_w_rec_check(img_path, diff_threshold, output_dir=svg_output_dir, config=svg_config, enforce_overwrite=True, topk=topk_paths)
    else:
        raise ValueError(f"method {method} not supported.")
    return svg_strs, svg_paths


def resize_image(input_path, output_path, new_width=None, new_height=None):
    with Image.open(input_path) as img:
        # Check if the image has an alpha channel
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            target_mode = 'RGBA'
        else:
            target_mode = 'RGB'
        
        img = img.convert(target_mode)
        width, height = img.size

        # Calculate new dimensions
        if new_width is not None:
            new_height = int((new_width / width) * height)
        elif new_height is not None:
            new_width = int((new_height / height) * width)
        else:
            raise ValueError("Either new_width or new_height must be specified")

        if new_width == width and new_height == height:
            print("No resizing needed")
            return input_path
        else:
            # Resizing the image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            # Save or display the resized image
            resized_img.save(output_path)
            return output_path
