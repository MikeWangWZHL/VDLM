import vtracer
from PIL import Image, ImageEnhance, ImageFilter
import os
from svg.path import parse_path
import numpy as np

def image_preprocessing(input_path, 
                        output_path, 
                        options={},
    ):
    img = Image.open(input_path)
    for option, factor in options.items():
        if option == "contrast":
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        elif option == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=factor))
        else:
            raise ValueError(f"Unknown option: {option}")
    img.save(output_path)

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

def is_significant(p1, p2, threshold=5):
    return np.sqrt((p2.real - p1.real)**2 + (p2.imag - p1.imag)**2) > threshold

def simplify_svg_path(path_data, threshold=5, round_to=0):
    path = parse_path(path_data)
    simplified_points = [path[0].start]

    for segment in path:
        if is_significant(simplified_points[-1], segment.end, threshold=threshold):
            simplified_points.append(segment.end)
    
    simplified_points = simplified_points[:-1]
    if len(simplified_points) < 3:
        return None
    else:
        new_path_data = 'M' + ' L'.join(f'{p.real:.0f},{p.imag:.0f}' for p in simplified_points) + " Z"
        return new_path_data

def post_processing(svg_file, output_path=None, min_area_ratio=None, simplify_threshold=None, rescale_width=None):
    from xml.etree import ElementTree as ET
    import re
    from shapely.geometry import Polygon
    import matplotlib.pyplot as plt
    import numpy as np
    from xml.dom import minidom
    from xml.dom.minidom import Document


    def get_image_size(dom):
        svg = dom.getElementsByTagName('svg')[0]
        width = svg.getAttribute('width')
        height = svg.getAttribute('height')
        return (int(width), int(height))

    def rescale_to_width(dom, new_width):
        # rescale to a specified width; keeping the aspect ratio
        ori_width, ori_height = get_image_size(dom)
        if ori_width == new_width:
            return dom
        else:
            scale_factor = new_width / ori_width
            print("Rescaling SVG to width:", new_width, "with scale factor:", scale_factor)
            svg = dom.getElementsByTagName('svg')[0]
            svg.setAttribute('width', str(new_width))
            svg.setAttribute('height', str(int(ori_height * scale_factor)))

            for path in dom.getElementsByTagName('path'):
                d = path.getAttribute('d')
                # set new d attribute
                new_d = re.sub(r"[-+]?\d*\.\d+|\d+", lambda m: str(int(int(m.group()) * scale_factor)), d)
                path.setAttribute('d', new_d)
                # set new transform attribute
                transform = path.getAttribute('transform')
                new_transform = re.sub(r"translate\(([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\)", 
                                    lambda m: f"translate({int(int(m.group(1)) * scale_factor)}, {int(int(m.group(2)) * scale_factor)})", 
                                    transform)
                path.setAttribute('transform', new_transform)
            return dom

    # Function to extract polygon points from path 'd' attribute in SVG
    def extract_polygon_points(path_d):
        return [(s.end.real, s.end.imag) for s in parse_path(path_d)]

    def calculate_polygon_area(points):
        return Polygon(points).area

    svg_data = open(svg_file).read()

    svg_data = svg_data.replace("<!-- Generator: visioncortex VTracer 0.6.0 -->", "")

    # Parsing the placeholder SVG data
    dom = minidom.parseString(svg_data)

    # rescale SVG
    if rescale_width is not None:
        dom = rescale_to_width(dom, rescale_width)

    # Get the width and height of the SVG
    width, height = get_image_size(dom)

    # Minimum area threshold
    if min_area_ratio is None:
        min_area_threshold = 0
    elif min_area_ratio < 1:
        min_area_threshold = max(0, width * height * min_area_ratio)
    else:
        min_area_threshold = min_area_ratio

    # Create a new SVG document
    new_doc = Document()
    svg = new_doc.createElement('svg')
    svg.setAttribute('xmlns', "http://www.w3.org/2000/svg")
    svg.setAttribute('version', "1.1")
    svg.setAttribute('width', str(width))
    svg.setAttribute('height', str(height))
    new_doc.appendChild(svg)

    # Add paths that meet the area criteria
    paths = dom.getElementsByTagName('path')

    # Shorten paths if required
    if simplify_threshold is not None:
        new_paths = []
        for path in paths:
            d = path.getAttribute('d')
            new_d = simplify_svg_path(d, threshold=simplify_threshold)
            if new_d is not None:
                path.setAttribute('d', new_d)
                new_paths.append(path)
    else:
        new_paths = paths

    # Add paths that meet the area criteria
    rm_path_count = 0
    for path in new_paths:
        d = path.getAttribute('d')
        try:
            points = extract_polygon_points(d)
            area = calculate_polygon_area(points)
            if area >= min_area_threshold:
                svg.appendChild(path)
            else:
                rm_path_count += 1
                # print("Removing path:", d)
        except:
            # print("ERROR: Error occured computing polygon for path:", d, "remove!")
            rm_path_count += 1
    
    filtered_svg_str = new_doc.toprettyxml(indent="")
    # filtered_svg_str = new_doc.toxml()
    # save to file
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(filtered_svg_str)


