import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import os
import math
from scipy.spatial import distance
from collections import defaultdict
from shapely.geometry import LineString, Point
import json
from tqdm import tqdm
import time
import sys
import os
from scipy.ndimage import binary_dilation, zoom, label

SVG_PROCESSOR_DIR= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image-to-SVG")
sys.path.append(SVG_PROCESSOR_DIR)

from get_svg import img_to_svg_str, check_svg_path_num
from configs import CONFIGS

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

## helper functions

# Kruskal's algorithm helper functions
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def kruskals(graph, vertices):
    result = []  # Store the resultant MST
    i, e = 0, 0  # Sorting edges by weight
    graph = sorted(graph, key=lambda item: item[2])
    parent, rank = [], []
    for node in range(vertices):
        parent.append(node)
        rank.append(0)
    while e < vertices - 1:
        u, v, w = graph[i]
        i = i + 1
        x = find(parent, u)
        y = find(parent, v)
        if x != y:
            e = e + 1
            result.append((u, v, w))
            union(parent, rank, x, y)
    return result

# Define a function to generate a random color
def random_color(alpha=1.0, min_brightness=0):
    """
    Generate a random color with a brightness threshold.
    
    :param alpha: Opacity of the color, defaults to 0.8
    :param min_brightness: Minimum brightness of the color
    :return: A tuple representing an RGBA color
    """
    while True:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        # Calculate the average to approximate brightness
        brightness = (r + g + b) / 3
        
        # Check if the color is bright enough
        if brightness >= min_brightness:
            return (r, g, b, int(255 * alpha))

# Function to rotate a point around another point
def rotate_point(cx, cy, angle, px, py):
    s = math.sin(math.radians(angle))
    c = math.cos(math.radians(angle))

    # translate point back to origin:
    px -= cx
    py -= cy

    # rotate point
    xnew = px * c - py * s
    ynew = px * s + py * c

    # translate point back:
    x = xnew + cx
    y = ynew + cy
    return x, y

# Function to calculate the new bounding box after rotation
def calculate_rotated_bbox(bbox, angle):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    corners = [
        (bbox[0], bbox[1]),
        (bbox[2], bbox[1]),
        (bbox[2], bbox[3]),
        (bbox[0], bbox[3])
    ]

    rotated_corners = [rotate_point(cx, cy, angle, *corner) for corner in corners]

    xs = [corner[0] for corner in rotated_corners]
    ys = [corner[1] for corner in rotated_corners]

    new_bbox = [min(xs), min(ys), max(xs), max(ys)]

    return rotated_corners, new_bbox

def check_corners_in_image(corners, img_size, tolerance=0):
    w, h = img_size
    for corner in corners:
        if corner[0] < -tolerance or corner[0] - w > tolerance or corner[1] < -tolerance or corner[1] - h > tolerance:
            return False
    return True

def calculate_bbox_area(bbox):
    return abs((bbox[2] - bbox[0])) * abs((bbox[3] - bbox[1]))

def polygon_area(coords):
    """
    Calculate the area of a polygon given its vertices.
    
    Parameters:
    coords (list of tuples): A list of (x, y) coordinates of the polygon's vertices.
    
    Returns:
    float: The area of the polygon.
    """
    n = len(coords)  # Number of vertices
    area = 0.0
    
    # Sum over the vertices
    for i in range(n):
        j = (i + 1) % n  # Next vertex index, wrapping around using modulo
        area += coords[i][0] * coords[j][1]
        area -= coords[i][1] * coords[j][0]
    ret = abs(area) / 2.0
    return ret

from math import acos, degrees, sqrt
def calculate_angle(AB, AC):
    
    # Calculate the dot product
    dot_product = AB[0] * AC[0] + AB[1] * AC[1]
    
    # Calculate the magnitudes of the vectors
    magnitude_AB = sqrt(AB[0]**2 + AB[1]**2)
    magnitude_AC = sqrt(AC[0]**2 + AC[1]**2)
    
    # Calculate the cosine of the angle between AB and BC
    cos_theta = dot_product / (magnitude_AB * magnitude_AC)
    
    # Calculate the angle in degrees
    angle = degrees(acos(cos_theta))
    
    return angle


def get_random_point_range(w, h):
    # randomly choose from the following areas
    ranges = [
        [(0, w), (0, h)], # whole image
        [(0, w), (0, h)], # whole image
        [(0, w), (0, h)], # whole image
        [(0, w), (0, h)], # whole image
        [(0, w//2), (0, h//2)], # left-top
        [(w//2, w), (0, h//2)], # right-top
        [(0, w//2), (h//2, h)], # left-bottom
        [(w//2, w), (h//2, h)], # right-bottom
        [(w//4, 3*w//4), (h//4, 3*h//4)], # center
        [(0, w), (0, h//2)], # horizontal 0 
        [(0, w), (h//4, 3*h//4)], # horizontal 1 
        [(0, w), (h//2, h)], # horizontal 2
        [(0, w//2), (0, h)], # vertical 0
        [(w//4, 3*w//4), (0, h)], # vertical 1
        [(w//2, w), (0, h)], # vertical 2
    ]
    return random.choice(ranges)

def validate_line_length_versus_width(p1, p2, width, threshold=5):
    # validate line length versus width
    line_length = distance.euclidean(p1, p2)
    # print("line_length, width, width * threshold", line_length, width, width * threshold)
    if line_length < width * threshold:
        return False
    return True

# Function to draw a point at a vertex
def draw_point_at_vertex(draw, point, fill, size=10):
    # Define the bounding box for the point
    x, y = point
    bounding_box = [x - size / 2, y - size / 2, x + size / 2, y + size / 2]
    draw.ellipse(bounding_box, fill=fill)


def get_non_transparent_bbox(img):
    data = np.array(img)
    non_transparent_indices = np.where(data[:, :, -1] > 0)
    x_min = np.min(non_transparent_indices[1])
    x_max = np.max(non_transparent_indices[1])
    y_min = np.min(non_transparent_indices[0])
    y_max = np.max(non_transparent_indices[0])
    return (x_min, y_min, x_max, y_max)

def count_obj_num(img):
    # Convert the image to a NumPy array and extract the alpha channel
    data = np.array(img)

    # Create a binary mask (1 for non-transparent pixels, 0 for transparent)
    binary_mask = np.where(data[:, :, 3] > 0, 1, 0)

    # Perform connected component analysis
    labeled_array, num_features = label(binary_mask)

    return num_features, binary_mask

def do_bounding_boxes_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.
    Each bounding box is defined as a tuple (x_min, y_min, x_max, y_max).
    
    Parameters:
    - box1: Tuple[int, int, int, int]
    - box2: Tuple[int, int, int, int]
    
    Returns:
    - bool: True if boxes overlap, False otherwise.
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Check if box1 is to the right of box2 or box2 is to the right of box1
    if x_min1 >= x_max2 or x_min2 >= x_max1:
        return False

    # Check if box1 is above box2 or box2 is above box1
    if y_min1 >= y_max2 or y_min2 >= y_max1:
        return False

    # If none of the above, the boxes overlap
    return True

def check_cur_contains_prev(cur_binary_mask, prev_binary_mask):
    # Create inverse mask of current binary mask
    inverse_cur_binary_mask = 1 - cur_binary_mask
    
    # Subtract previous binary mask from the inverse of current binary mask
    subtraction_result = inverse_cur_binary_mask - prev_binary_mask
    
    # If the subtraction result has only zeros, current object contains the previous one
    if_cur_contains_prev = np.all(subtraction_result <= 0)
    
    return if_cur_contains_prev

def calculate_iou(maskA, maskB):
    """
    Calculate the Intersection over Union (IOU) of two binary masks.
    
    Parameters:
    - maskA: numpy array of the first mask.
    - maskB: numpy array of the second mask.
    
    Returns:
    - IOU value as a float.
    """
    intersection = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()
    iou = intersection / union
    return iou

def intersection_ratio(maskA, maskB):
    """
        output the intersection/self_area ratio of two binary masks
    """
    intersection = np.logical_and(maskA, maskB).sum()
    intersection_ratio_A = intersection / maskA.sum()
    intersection_ratio_B = intersection / maskB.sum()
    return intersection_ratio_A, intersection_ratio_B

def remove_duplicated_edges(edges):
    new_edges = []
    added = set()
    for edge in edges:
        if edge not in added:
            new_edges.append(edge)
            added.add(edge)
            added.add((edge[1], edge[0]))
    return new_edges

def point_selection_with_constraints(reference_objects, point_range, p = 0.5):
    if reference_objects is None or len(reference_objects) == 0:
        return (random.randint(*point_range[0]), random.randint(*point_range[1]))
    
    if random.random() < p:
        candidate_points = []
        # randomly pick an obj
        obj = random.choice(reference_objects)
        if obj["type"] == "circle":
            center = obj["center"]
            radius = obj["radius"]
            theta = random.uniform(0, 2 * math.pi)
            random_point_on_circle = (center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta))
            candidate_points += [center, random_point_on_circle]
        elif obj["type"] == "line_segment":
            vertices = obj["vertices"]
            center_of_line = ((vertices[0][0] + vertices[1][0]) / 2.0, (vertices[0][1] + vertices[1][1]) / 2.0)
            t = random.random()
            random_point_on_line = (vertices[0][0] + t * (vertices[1][0] - vertices[0][0]), vertices[0][1] + t * (vertices[1][1] - vertices[0][1]))
            candidate_points += [vertices[0], vertices[1], random_point_on_line, center_of_line]
        elif obj["type"] == "rectangle":
            vertices = obj["rotated_vertices"]
            center_of_rectangle = (sum(vertex[0] for vertex in vertices) / 4.0, sum(vertex[1] for vertex in vertices) / 4.0)
            middle_points = [((vertices[i][0] + vertices[(i + 1) % 4][0]) / 2.0, (vertices[i][1] + vertices[(i + 1) % 4][1]) / 2.0) for i in range(4)]
            candidate_points += vertices
            candidate_points += middle_points
            candidate_points.append(center_of_rectangle)
        elif obj["type"] == "triangle":
            vertices = obj["vertices"]
            center_of_triangle = (sum(vertex[0] for vertex in vertices) / 3.0, sum(vertex[1] for vertex in vertices) / 3.0)
            middle_points = [((vertices[i][0] + vertices[(i + 1) % 3][0]) / 2.0, (vertices[i][1] + vertices[(i + 1) % 3][1]) / 2.0) for i in range(3)]
            candidate_points += vertices
            candidate_points += middle_points
            candidate_points.append(center_of_triangle)
        else:
            return (random.randint(*point_range[0]), random.randint(*point_range[1]))
        return random.choice(candidate_points)
    else:
        return (random.randint(*point_range[0]), random.randint(*point_range[1]))

def length_selection_with_constraints(reference_objects, length_range, p=0.5):
    if reference_objects is None or len(reference_objects) == 0:
        return random.randint(*length_range)
    
    if random.random() < p:
        # randomly pick an obj
        obj = random.choice(reference_objects)
        if obj["type"] == "circle":
            radius = obj["radius"]
            # Options: radius, diameter, 
            length_options = [radius, 2 * radius]
        elif obj["type"] == "line_segment":
            vertices = obj["vertices"]
            # Length of the segment
            length = math.sqrt((vertices[1][0] - vertices[0][0])**2 + (vertices[1][1] - vertices[0][1])**2)
            length_options = [length, length / 2]
        elif obj["type"] == "rectangle":
            vertices = obj["rotated_vertices"]
            # Side lengths and diagonal
            side1 = math.sqrt((vertices[1][0] - vertices[0][0])**2 + (vertices[1][1] - vertices[0][1])**2)
            side2 = math.sqrt((vertices[2][0] - vertices[1][0])**2 + (vertices[2][1] - vertices[1][1])**2)
            diagonal = math.sqrt((vertices[2][0] - vertices[0][0])**2 + (vertices[2][1] - vertices[0][1])**2)
            length_options = [side1, side2, diagonal, diagonal / 2]
        elif obj["type"] == "triangle":
            vertices = obj["vertices"]
            # Side lengths
            side1 = math.sqrt((vertices[1][0] - vertices[0][0])**2 + (vertices[1][1] - vertices[0][1])**2)
            side2 = math.sqrt((vertices[2][0] - vertices[1][0])**2 + (vertices[2][1] - vertices[1][1])**2)
            side3 = math.sqrt((vertices[2][0] - vertices[0][0])**2 + (vertices[2][1] - vertices[0][1])**2)
            
            centroid_x = (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3
            centroid_y = (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3
            
            j = random.randint(0, 2)
            centroid_to_random_vertex_distance = math.sqrt((centroid_x - vertices[j][0])**2 + (centroid_y - vertices[j][1])**2)
        
            length_options = [side1, side2, side3, centroid_to_random_vertex_distance]
        else:
            # Default case if no suitable type is found
            return random.randint(*length_range)  # Or some default length
        return random.choice(length_options)
    else:
        return random.randint(*length_range)

def check_if_duplicate(obj, reference_objs):
    for ref_obj in reference_objs:
        if ref_obj["type"] == obj["type"]:
            if ref_obj["type"] == "circle":
                if ref_obj["center"] == obj["center"] and ref_obj["radius"] == obj["radius"]:
                    return True
            elif ref_obj["type"] == "line_segment":
                if sorted(ref_obj["vertices"]) == sorted(obj["vertices"]):
                    return True
            elif ref_obj["type"] == "rectangle":
                if sorted(ref_obj["rotated_vertices"]) == sorted(obj["rotated_vertices"]) or sorted(ref_obj["unrotated_vertices"]) == sorted(obj["unrotated_vertices"]):
                    return True
            elif ref_obj["type"] == "triangle":
                if sorted(ref_obj["vertices"]) == sorted(obj["vertices"]):
                    return True
            else:
                raise ValueError(f"object type {ref_obj["type"]} not implemented")
    return False

# Define a function to generate random shapes with annotations
DEFAULT_SHAPE_RANGE_CONFIGS = {
    "path": [3, 16],
    "polygon": [5, 20],
    "grid": [2, 6],
    "graph": [4, 16],
}
def generate_shape(
        img,
        shape_type, 
        img_size,
        style="filled", # ["filled", "outline", "fill_outline"]
        alpha = 1.0,
        enforce_fill_color = None,
        enforce_outline_color = None,
        enforce_point_range = None,
        shape_range_configs = DEFAULT_SHAPE_RANGE_CONFIGS,
        select_shape_range_w_decreasing_prob = False,
        reference_objects = None,
        constraint_prob = 0.5,
    ):
    w, h = img_size
    point_range = get_random_point_range(w, h) if enforce_point_range is None else enforce_point_range

    ## set suitable line width range ##
    line_width_mult = max(w / 512, h / 512, 1)

    # small regoin
    if point_range[0][1] - point_range[0][0] <= w // 2 and point_range[1][1] - point_range[1][0] <= h // 2:
        min_width, max_width = 2, 5
    # middle region
    elif point_range[0][1] - point_range[0][0] <= w // 2 or point_range[1][1] - point_range[1][0] <= h // 2:
        min_width, max_width = 2, 6
    # large region
    else:
        min_width, max_width = 2, 7

    if shape_type in ["circle", "ellipse", "triangle", "polygon", "rectangle", "pieslice", "chord"]:
        min_width, max_width = 2, 5

    if style == "filled":
        fill = random_color(alpha) if enforce_fill_color is None else enforce_fill_color
        outline = None
        width = 1
    elif style == "outline":
        outline = random_color(alpha) if enforce_outline_color is None else enforce_outline_color
        fill = None
        width = random.randint(min_width, max_width) * line_width_mult
    elif style == "fill_outline":
        fill = random_color(alpha) if enforce_fill_color is None else enforce_fill_color
        outline = random_color(alpha) if enforce_outline_color is None else enforce_outline_color
        width = random.randint(min_width, max_width) * line_width_mult

    # outline only shapes
    if shape_type in ["path", "grid", "graph", "line_segment", "arc"]:
        fill = outline = random_color(alpha=1.0) if enforce_outline_color is None else enforce_outline_color
        width = random.randint(min_width, max_width) * line_width_mult

    width = min(int(width), 10)
    #####
    
    # joint style for draw.line
    line_joint_styles = ["curve", None, "point"]
    line_joint_styles_probs = [0.4, 0.4, 0.2]
    line_joint = np.random.choice(line_joint_styles, p=line_joint_styles_probs)
    point_size = width * random.uniform(1, 2)

    # Generate a random rotation angle
    angle = random.randint(0, 180)

    # new object disentangled image
    cur_obj_img = Image.new("RGBA", img_size, (0, 0, 0, 0))
    cur_obj_img_draw = ImageDraw.Draw(cur_obj_img)

    if shape_type == "circle":
        if style == "outline":
            radius_range = (width*5, min(w, h)//2 - 10)
        elif style == "filled":
            radius_range = (5, min(w, h)//2 - 10)
        else:    
            radius_range = (width*5, min(w, h)//2 - 10)
        if reference_objects is not None:
            while True:
                center = point_selection_with_constraints(reference_objects, point_range, p=constraint_prob)
                cx, cy = center
                radius = min(length_selection_with_constraints(reference_objects, radius_range, p=constraint_prob), min(cx, cy, w-cx, h-cy))
                if not check_if_duplicate({"type": "circle", "center":center, "radius":radius}, reference_objects):
                    break
        else:
            radius = random.randint(*radius_range)
            center = (random.randint(radius, w-radius), random.randint(radius, h-radius))
        bounding_box = [center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius]
        cur_obj_img_draw.ellipse(bounding_box, fill=fill, outline=outline, width=width)
        # Paste the rotated ellipse back onto the main image canvas
        ret_ann = {"type": "circle", "bbox":bounding_box, "center":center, "radius":radius, "style":style, "fill": fill, "outline": outline, "width": width}
    
    elif shape_type == "ellipse":
        max_attempts = 10000
        attempts = 0
        while True:
            # Generate a random bounding box for the ellipse
            major_axis = random.randint(20, min(w, h) // 2)
            minor_axis = random.randint(10, min(major_axis, w // 4, h // 4))
            center = (random.randint(major_axis, w-major_axis), random.randint(minor_axis, h-major_axis))
            bounding_box = [center[0]-major_axis, center[1]-minor_axis, center[0]+major_axis, center[1]+minor_axis]
            corners = [(bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[1]), (bounding_box[2], bounding_box[3]), (bounding_box[0], bounding_box[3])]
            # rotated bounding box
            rotated_corners, _ = calculate_rotated_bbox(bounding_box, -angle) # corner rotates in the opposite direction
            # check if axis is not too small compared to the line width, and the corners are in the image
            if check_corners_in_image(rotated_corners, img_size, tolerance = 10) and validate_line_length_versus_width(bounding_box[0:2], bounding_box[2:4], width, threshold=10):
                break
            attempts += 1
            if attempts > max_attempts:
                print("Ellipse break due to max attempts...")
                break

        cur_obj_img_draw.ellipse(bounding_box, fill=fill, outline=outline, width=width)
        # Rotate the ellipse image
        cur_obj_img = cur_obj_img.rotate(angle, center=center)
        ret_ann = {"type": "ellipse", "unrotated_corners":corners, "rotated_corners": rotated_corners, "center": center, "major_axis": major_axis, "minor_axis": minor_axis, "angle": angle, "style": style, "fill": fill, "outline": outline, "width": width}

    elif shape_type == "rectangle":
        max_attempts = 10000
        attempts = 0
        while True:
            # Generate a random size for the rectangle
            if reference_objects is not None:
                while True:
                    point_1 = point_selection_with_constraints(reference_objects, point_range, p=constraint_prob)
                    point_2 = point_selection_with_constraints(reference_objects, point_range, p=constraint_prob)
                    if point_1[0] != point_2[0] and point_1[1] != point_2[1]:
                        break
                top_left_x, top_left_y = min(point_1[0], point_2[0]), min(point_1[1], point_2[1])
                rect_width = max(abs(point_1[0] - point_2[0]), 20)
                rect_height = max(abs(point_1[1] - point_2[1]), 20)
            else:
                rect_width = random.randint(20, w // 1.5)
                rect_height = random.randint(20, h // 1.5)
                top_left_x = random.randint(5, w - rect_width)
                top_left_y = random.randint(5, h - rect_height)
            
            bounding_box = [top_left_x, top_left_y, top_left_x + rect_width, top_left_y + rect_height]
            corners = [(bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[1]), (bounding_box[2], bounding_box[3]), (bounding_box[0], bounding_box[3])]
            center = ((bounding_box[0] + bounding_box[2]) / 2, (bounding_box[1] + bounding_box[3]) / 2)

            # Calculate the new bounding box and rotated corners after rotation
            rotated_corners, rotated_bounding_box = calculate_rotated_bbox(bounding_box, -angle) # Corner rotates in the opposite direction
            
            if check_corners_in_image(rotated_corners, img_size, tolerance = 10) and validate_line_length_versus_width(bounding_box[0:2], bounding_box[2:4], width, threshold=10):
                if reference_objects is None:
                    break
                else:
                    if not check_if_duplicate({"type": "rectangle", "rotated_vertices":rotated_corners, "unrotated_vertices": corners}, reference_objects):
                        break
            
            attempts += 1
            if attempts > max_attempts:
                print("Rectangle break due to max attempts...")
                break
        
            
        cur_obj_img_draw.rectangle(bounding_box, fill=fill, outline=outline, width=width)
        if reference_objects is not None:
            rotation_p = 0.2
        else:
            rotation_p = 0.8
        if random.random() < rotation_p:
            cur_obj_img = cur_obj_img.rotate(angle, center=center)
            ret_ann = {"type": "rectangle", "unrotated_vertices":corners, "rotated_vertices": rotated_corners, "unrotated_bounding_box": bounding_box, "rotated_bounding_box": rotated_bounding_box, "angle": angle, "style": style, "fill": fill, "outline": outline, "width": width}
        else:
            ret_ann = {"type": "rectangle", "unrotated_vertices":corners, "rotated_vertices": corners, "unrotated_bounding_box": bounding_box, "rotated_bounding_box": bounding_box, "angle": 0, "style": style, "fill": fill, "outline": outline, "width": width}

    elif shape_type == "triangle":
        max_attempts = 10000
        attempts = 0
        while True:
            # Generate three random points for the triangle
            if reference_objects is not None:
                while True:
                    point_1 = point_selection_with_constraints(reference_objects, point_range, p=constraint_prob)
                    point_2 = point_selection_with_constraints(reference_objects, point_range, p=constraint_prob)
                    point_3 = point_selection_with_constraints(reference_objects, point_range, p=constraint_prob)
                    points = [point_1, point_2, point_3]
                    if point_1 != point_2 and point_2 != point_3 and point_1 != point_3:
                        if not check_if_duplicate({"type": "triangle", "vertices":points}, reference_objects):
                            break
            else:
                points = [(random.randint(*point_range[0]), random.randint(*point_range[1])) for _ in range(3)]
            # check if the lines are not too near the same line
            if polygon_area(points) > 200 * (width / 3):
                break
            attempts += 1
            if attempts > max_attempts:
                print("Triangle break due to max attempts...")
                break
        cur_obj_img_draw.polygon(points, fill=fill, outline=outline, width=width)
        ret_ann = {"type": "triangle", "vertices": points, "style": style, "fill": fill, "outline": outline, "width": width}
    
    elif shape_type == "polygon":
        if select_shape_range_w_decreasing_prob:
            r = shape_range_configs["polygon"]
            values = [i for i in range(r[0], r[1]+1, 1)]
            weights = np.arange(len(values), 0, -1)
            num_vertices = np.random.choice(values, p=weights/np.sum(weights))
        else:
            num_vertices = random.randint(*shape_range_configs["polygon"])
        
        max_attempts = 10000
        attempts = 0
        while True:
            # points = [(random.randint(*point_range[0]), random.randint(*point_range[1])) for _ in range(num_vertices)]
            points = []
            # iteratively generate random points
            for _ in range(num_vertices):
                tries = 100
                while True:
                    p = (random.randint(*point_range[0]), random.randint(*point_range[1]))
                    if tries == 0:
                        print("Failed to generate a good point after 100 tries, pick a random point...")
                        points.append(p)
                        break
                    if points == []:
                        points.append(p)
                        break
                    else:
                        # check distance to existing points
                        valid = True
                        for i in range(len(points)):
                            if not validate_line_length_versus_width(points[i], p, width, threshold = 5):
                                valid = False
                                break
                        if valid:
                            points.append(p)
                            break
                    tries -= 1

            # Calculate the centroid of these points
            centroid = (sum(x for x, _ in points) / num_vertices, sum(y for _, y in points) / num_vertices)
            # Sort the points by angle relative to the centroid
            sorted_points = sorted(points, key=lambda point: np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))
            
            # check if area is too small
            if polygon_area(sorted_points) > 200 * (width / 3):
                break
            attempts += 1
            if attempts > max_attempts:
                print("Polygon break due to max attempts...")
                break

        if style == "outline":
            draw_points = sorted_points + [sorted_points[0]]
            cur_obj_img_draw.line(draw_points, fill=outline, width=width, joint=line_joint)
        else:
            cur_obj_img_draw.polygon(sorted_points, fill=fill, outline=outline, width=width)

        ret_ann = {"type": "polygon", "vertices": sorted_points, "style": style, "fill": fill, "outline": outline, "width": width, "joint": line_joint}
    
    elif shape_type == "line_segment":
        while True:

            if reference_objects is not None:
                while True:
                    point_1 = point_selection_with_constraints(reference_objects, point_range, p=constraint_prob)
                    point_2 = point_selection_with_constraints(reference_objects, point_range, p=constraint_prob)
                    points = [point_1, point_2]
                    if point_1 != point_2:
                        if not check_if_duplicate({"type": "line_segment", "vertices":points}, reference_objects):
                            break
            else:
                points = [(random.randint(*point_range[0]), random.randint(*point_range[1])), (random.randint(*point_range[0]), random.randint(*point_range[1]))]
            
            
            if validate_line_length_versus_width(points[0], points[1], width, threshold=5):
                break
        if line_joint in ["curve", None]:
            cur_obj_img_draw.line(points, fill=fill, width=width, joint=line_joint)
        else:
            cur_obj_img_draw.line(points, fill=fill, width=width, joint=None)
            
            for p in points:
                draw_point_at_vertex(cur_obj_img_draw, p, fill, size=point_size)
        ret_ann = {"type": "line_segment", "vertices": points, "style": "outline", "fill": fill, "outline": outline, "width": width, "joint": line_joint}
    
    elif shape_type in ["arc", "pieslice", "chord"]:
        
        max_attempts = 10000
        attempts = 0
        while True:
            # Generate a random bounding box for the ellipse part of the arc
            major_axis = random.randint(20, min(w, h) // 2)
            minor_axis = random.randint(10, min(major_axis, w // 4, h // 4))
            center = (random.randint(major_axis, w-major_axis), random.randint(minor_axis, h-major_axis))
            bounding_box = [center[0]-major_axis, center[1]-minor_axis, center[0]+major_axis, center[1]+minor_axis]
            
            # Generate random start and end angles for the arc
            start_angle = random.randint(-180, 180)
            end_angle = random.randint(start_angle, start_angle + 360)

            # check arc length
            if 30 < end_angle - start_angle < 330 and calculate_bbox_area(bounding_box) > 200 * (width / 3) and validate_line_length_versus_width(bounding_box[0:2], bounding_box[2:4], width, threshold=15):
                break
            attempts += 1
            if attempts > max_attempts:
                print("break due to max attempts...")
                break

        if shape_type == "arc":
            # Draw the arc
            cur_obj_img_draw.arc(bounding_box, start_angle, end_angle, fill=fill, width=width)
            ret_ann = {"type": "arc", "bounding_box": bounding_box, "start_angle": start_angle, "end_angle": end_angle, "style": "outline", "fill": fill, "outline": outline, "width": width}
        elif shape_type == "pieslice":
            # Draw the pieslice
            cur_obj_img_draw.pieslice(bounding_box, start_angle, end_angle, fill=fill, outline=outline, width=width)
            ret_ann = {"type": "pieslice", "bounding_box": bounding_box, "start_angle": start_angle, "end_angle": end_angle, "style": style, "fill": fill, "outline": outline, "width": width}
        elif shape_type == "chord":
            # Draw the chord
            cur_obj_img_draw.chord(bounding_box, start_angle, end_angle, fill=fill, outline=outline, width=width)
            ret_ann = {"type": "chord", "bounding_box": bounding_box, "start_angle": start_angle, "end_angle": end_angle, "style": style, "fill": fill, "outline": outline, "width": width}

    elif shape_type == "path": # non-intersecting path
        if select_shape_range_w_decreasing_prob:
            r = shape_range_configs["path"]
            values = [i for i in range(r[0], r[1]+1, 1)]
            weights = np.arange(len(values), 0, -1)
            num_vertices = np.random.choice(values, p=weights/np.sum(weights))
        else:
            num_points = random.randint(*shape_range_configs["path"])  # Choose a random number of points for the path
        
        points = [(random.randint(*point_range[0]), random.randint(*point_range[1])) for _ in range(3)]
        
        max_attempts = 10000
        attempts = 0

        distance_threshold = width * 2

        while len(points) < num_points:
            
            if attempts > max_attempts:
                print("Path break due to max attempts...")
                break

            new_point = (random.randint(*point_range[0]), random.randint(*point_range[1]))

            intersects = False
            too_close_to_existing_lines = False
            new_line = LineString([points[-1], new_point])
            for i in range(1, len(points)-1):
                existing_line = LineString([points[i-1], points[i]])
                if existing_line.intersects(new_line):
                    intersects = True
                    break
                # new point to existing line distance
                point_to_line_dist = abs(existing_line.distance(Point(new_point)))
                if abs(point_to_line_dist) < distance_threshold:
                    too_close_to_existing_lines = True
                    break
                # existing point to new line distance
                point_to_line_dist = abs(new_line.distance(Point(points[i-1])))
                if abs(point_to_line_dist) < distance_threshold:
                    too_close_to_existing_lines = True
                    break

            if not intersects and not too_close_to_existing_lines:
                points.append(new_point)
            else:
                attempts += 1
        if line_joint in ["curve", None]:
            # Draw the non-intersecting path
            cur_obj_img_draw.line(points, fill=fill, width=width, joint=line_joint)
        else:
            cur_obj_img_draw.line(points, fill=fill, width=width, joint=None)
            for p in points:
                draw_point_at_vertex(cur_obj_img_draw, p, fill, size=point_size)
        ret_ann = {"type": "path", "vertices": points, "style": "outline", "fill": fill, "outline": outline, "width": width, "joint": line_joint}

    elif shape_type == "grid":
        max_grid_n, min_grid_n = shape_range_configs["grid"] # [2, 6]
        grid_size_w = random.randint( w // min_grid_n, w // max_grid_n)  # Define the size of the grid
        grid_size_h = random.randint( h // min_grid_n, h // max_grid_n)  # Define the size of the grid
        num_x = (w - 10) // grid_size_w  # Number of points horizontally
        num_y = (h - 10) // grid_size_h  # Number of points vertically

        # Generate grid points
        padding_x, padding_y = (w - num_x * grid_size_w) // 2, (h - num_y * grid_size_h) // 2
        points = [(x * grid_size_w + padding_x, y * grid_size_h + padding_y) for x in range(0, num_x + 1) for y in range(0, num_y + 1)]

        visited = set()  # Keep track of visited points
        paths = []  # Store paths to draw

        # Function to get adjacent points
        def get_adjacent_dfs(point):
            x, y = point
            adjacent_points = [(x - grid_size_w, y), (x + grid_size_w, y), (x, y - grid_size_h), (x, y + grid_size_h)]
            # Filter out points outside the grid or already visited
            return [p for p in adjacent_points if p in points and p not in visited]

        # Start from a random point in the grid
        current_point = random.choice(points)
        stack = [current_point]
        visited.add(current_point)

        # Iteratively connect points until all points are visited
        # DFS
        while stack:
            current_point = stack[-1]
            adj_points = get_adjacent_dfs(current_point)

            if adj_points:
                next_point = random.choice(adj_points)
                paths.append((current_point, next_point))
                visited.add(next_point)
                stack.append(next_point)
            else:
                stack.pop()

        # Draw DFS paths
        paths = sorted(paths)
        for path in paths:
            cur_obj_img_draw.line(path, fill=fill, width=width, joint=line_joint)
        
        existing_paths = set(paths)
        if random.random() < 0.5:
            # draw additional edges
            remaining_edges = []
            added = set()
            for point in points:
                x, y = point
                adj_points = [(x - grid_size_w, y), (x + grid_size_w, y), (x, y - grid_size_h), (x, y + grid_size_h)]
                adj_points = [p for p in adj_points if p in points]
                for adj_p in adj_points:
                    if (point, adj_p) not in existing_paths and (adj_p, point) not in existing_paths and (point, adj_p) not in added and (adj_p, point) not in added:
                        remaining_edges.append((point, adj_p))
                        added.add((point, adj_p))
                        added.add((adj_p, point))
            if len(remaining_edges) > 0:
                num_additional_edges = random.randint(1, len(remaining_edges))  # Randomly decide how many additional edges to add, up to half of existing
                additional_graph_edges = random.sample(remaining_edges, num_additional_edges)
                for edge in additional_graph_edges:
                    assert edge not in paths
                    cur_obj_img_draw.line(edge, fill=fill, width=width, joint=line_joint)
                paths += additional_graph_edges
        
        edges = remove_duplicated_edges(paths)
        edges = [sorted(edge, key=lambda x: x[0] + x[1]) for edge in edges]
        edges = sorted(edges, key=lambda x: x[0][0] + x[0][1] + x[1][0] + x[1][1])

        ret_ann = {"type": "grid", "vertices": sorted(points), "edges": edges, "style": "outline", "fill": fill, "outline": outline, "width": width, "joint": line_joint, "num_points_x": num_x, "num_points_y": num_y}

    elif shape_type == "graph":
        def find_groups(u, visited, adjacency_list, group):
            # use recursive DFS to find connected components
            visited[u] = True
            group.append(u)
            for v in adjacency_list[u]:
                if not visited[v]:
                    find_groups(v, visited, adjacency_list, group)

        def draw_edges_in_groups(graph_edges):
            adjacency_list = defaultdict(list)
            for u, v, _ in graph_edges:
                adjacency_list[u].append(v)
                adjacency_list[v].append(u)
            # Find all groups of connected points
            visited = {node: False for node in adjacency_list}
            groups = []
            for u in adjacency_list:
                if not visited[u]:
                    group = []
                    find_groups(u, visited, adjacency_list, group)
                    groups.append(group)
            for group in groups:
                # For each group, draw a polyline connecting all points in the group
                polyline = [points[u] for u in group]
                cur_obj_img_draw.line(polyline, fill=fill, width=width, joint=line_joint)
            # get all edges
            edges = []
            for u, v, _ in graph_edges:
                edges.append((points[u], points[v]))
            return edges

        def draw_edges_individually(graph_edges):
            edges = []
            for edge in graph_edges:
                u, v, _ = edge
                cur_obj_img_draw.line([points[u], points[v]], fill=fill, width=width, joint=line_joint)
                edges.append((points[u], points[v]))
            return edges

        if select_shape_range_w_decreasing_prob:
            r = shape_range_configs["graph"]
            values = [i for i in range(r[0], r[1]+1, 1)]
            weights = np.arange(len(values), 0, -1)
            points_num = np.random.choice(values, p=weights/np.sum(weights))
        else:
            points_num = random.randint(*shape_range_configs["graph"])

        points = []
        # iteratively generate random points
        for _ in range(points_num):
            tries = 100
            while True:
                p = (random.randint(*point_range[0]), random.randint(*point_range[1]))
                if tries == 0:
                    print("Failed to generate a good point after 100 tries, pick a random point...")
                    points.append(p)
                    break
                if points == []:
                    points.append(p)
                    break
                else:
                    # check distance to existing points
                    valid = True
                    for i in range(len(points)):
                        if not validate_line_length_versus_width(points[i], p, width, threshold = 5):
                            valid = False
                            break
                    if valid:
                        points.append(p)
                        break
                tries -= 1

        assert len(points) == points_num
        # Calculate distances and create a graph
        graph = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = distance.euclidean(points[i], points[j])
                graph.append((i, j, dist))

        # Apply Kruskal's algorithm to find MST
        mst = kruskals(graph, len(points))

        # # Draw the MST one edge at a time
        edges = draw_edges_individually(mst)
        # edges = draw_edges_in_groups(mst)

        # for some probabilties, add random additional edges
        if random.random() < 0.5:
            remaining_graph_edges = [edge for edge in graph if edge not in mst]
            if len(remaining_graph_edges) > 0:
                num_additional_edges = random.randint(1, max(1, len(edges) // 3))  # Randomly decide how many additional edges to add, up to half of existing
                additional_graph_edges = random.sample(remaining_graph_edges, num_additional_edges)
                additional_edges = draw_edges_individually(additional_graph_edges)
                edges += additional_edges
        
        edges = remove_duplicated_edges(edges)
        edges = [sorted(edge, key=lambda x: x[0] + x[1]) for edge in edges]
        edges = sorted(edges, key=lambda x: x[0][0] + x[0][1] + x[1][0] + x[1][1])
        ret_ann = {"type": "graph", "vertices": sorted(points), "edges": edges, "style": "outline", "fill": fill, "outline": outline, "width": width, "joint": line_joint}

    # Paste the new object to the main image
    img.paste(cur_obj_img, (0, 0), cur_obj_img)
    return cur_obj_img, ret_ann

# resize pil image
def resize_image_(img, new_width=None, new_height=None):
    width, height = img.size
    # Calculate new dimensions
    if new_width is not None:
        new_height = int((new_width / width) * height)
    elif new_height is not None:
        new_width = int((new_height / height) * width)
    else:
        raise ValueError("Either new_width or new_height must be specified")
    
    if new_width == width and new_height == height:
        return img

    # Resizing the image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    return resized_img

def add_color_noise_rgba(img, noise_color=(255, 0, 0, 255), ratio=0.1, intensity_range=(0.5, 0.1), dilation_range=(2, 3), noise_size=2):
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    intensity = random.uniform(*intensity_range)

    # Convert PIL image to NumPy array
    img_array = np.array(img)

    # Identify non-transparent (alpha != 0) areas
    non_transparent_mask = img_array[:, :, 3] != 0

    # Dilate the non-transparent mask to include areas within the specified range
    dilation_dist = random.randint(*dilation_range)
    dilated_mask = binary_dilation(non_transparent_mask, structure=np.ones((dilation_dist, dilation_dist)))

    # Modify initial noise generation to control particle size
    if noise_size > 1:
        reduced_dims = (img_array.shape[0] // noise_size, img_array.shape[1] // noise_size)
        small_noise = np.random.rand(*reduced_dims) < ratio
        
        # Upsample the small noise to the original image size
        zoom_factors = (img_array.shape[0] / reduced_dims[0], img_array.shape[1] / reduced_dims[1])
        large_noise = zoom(small_noise.astype(float), zoom_factors, order=0) > 0.5  # Nearest-neighbor interpolation
    else:
        large_noise = np.random.rand(*img_array.shape[:2]) < ratio

    noise_mask = large_noise & dilated_mask

    # add noise to img_array
    for i in range(4):  # Apply to R, G, B, and A channels
        # For the alpha channel, adjust noise application logic as needed
        if i < 3:  # For RGB, use the color intensity
            img_array[:, :, i] = np.where(img_array[:, :, 3] > 0, img_array[:, :, i], noise_color[i] * intensity) # not changing the color within the object
        else:
            img_array[:, :, i] = np.where(img_array[:, :, 3] > 0, img_array[:, :, i], 255 * noise_mask)

    # Convert back to PIL image
    img_with_noise = Image.fromarray(img_array, 'RGBA')
    return img_with_noise

def img_augmentation_before_svg_conversion(img, data_augmentation):
    
    if "random_noise" in data_augmentation:
        if random.random() < data_augmentation["random_noise"]["prob"]:
            noise_color = data_augmentation["random_noise"].get("color", random_color())
            img = add_color_noise_rgba(
                img, 
                noise_color=noise_color, 
                ratio=data_augmentation["random_noise"].get("ratio", 0.1),
                intensity_range=data_augmentation["random_noise"].get("intensity_range", (0.5, 1)),
                dilation_range=data_augmentation["random_noise"].get("dilation_range", (2, 5)),
                noise_size=data_augmentation["random_noise"].get("noise_size", 1)
            )

    if "blur" in data_augmentation:
        if random.random() < data_augmentation["blur"]["prob"]:
            blur_radius_range = data_augmentation["blur"].get("radius", (0.1, 0.5))  # Default blur radius
            blur_radius = random.uniform(*blur_radius_range)
            img = img.filter(ImageFilter.GaussianBlur(blur_radius))
            # print(f"Applied Gaussian blur with radius {blur_radius}")

    if "random_resizing" in data_augmentation:
        if random.random() < data_augmentation["random_resizing"]["prob"]:
            cur_w, cur_h = img.size
            rand_width = random.randint(int(cur_w*0.8), int(cur_w*1.2))
            img = resize_image_(img, new_width=rand_width)
            print(f"random resizing from {cur_w}x{cur_h} to {img.size}")

    return img

def do_masks_contain(maskA, maskB):
    """
    Check if maskA contains maskB, or maskB contains maskA.
    
    Parameters:
    - maskA: numpy array of the first mask.
    - maskB: numpy array of the second mask.
    
    Returns:
    - True if maskA contains maskB, or maskB contains maskA.
    """
    # Perform logical AND between maskA and maskB
    overlap = np.logical_and(maskA, maskB)
    
    # Check if the overlap is identical to maskB (meaning maskA contains maskB)
    is_contained = np.array_equal(overlap, maskB) or np.array_equal(overlap, maskA)
    
    return is_contained

# Main function to generate dataset
def generate_dataset(
        num_images, 
        num_shapes_per_image = [1], 
        num_shapes_per_image_prob = None,
        shape_types = ["circle", "ellipse", "rectangle", "triangle", "polygon", "line_segment", "arc", "pieslice", "chord", "grid", "graph", "path"],
        shape_types_prob = None,
        shape_styles = ["filled", "outline", "fill_outline"],
        shape_styles_prob = [0.4, 0.4, 0.2],
        shape_range_configs = DEFAULT_SHAPE_RANGE_CONFIGS,
        select_shape_range_w_decreasing_prob = False,
        output_dir = None,
        output_prefix = "",
        save_background = False,
        svg_config = None,
        data_augmentation = None,
        enforce_single_svg_path = False,
        enforce_same_color = False,
        enforce_same_style = False,
        enforce_same_point_range = False,
        enforce_object_overlap = False,
        use_constraints = False
    ):
    if output_dir is not None:
        output_img_dir = os.path.join(output_dir, "images")
        os.makedirs(output_img_dir, exist_ok=True)
        if svg_config is not None:
            output_svg_dir = os.path.join(output_dir, "svg")
            os.makedirs(output_svg_dir, exist_ok=True)


    if shape_types_prob is None:
        shape_types_prob = [1/len(shape_types)] * len(shape_types)
    
    if num_shapes_per_image_prob is None:
        num_shapes_per_image_prob = [1/len(num_shapes_per_image)] * len(num_shapes_per_image)
    
    all_annotations = {
        "metadata":{
            "num_images": num_images,
            "num_shapes_per_image": num_shapes_per_image,
            "num_shapes_per_image_prob": num_shapes_per_image_prob,
            "shape_types": shape_types,
            "shape_types_prob": shape_types_prob,
            "shape_styles": shape_styles,
            "shape_styles_prob": shape_styles_prob,
        },
        "data": {}
    }
    for i in tqdm(range(num_images)):

        while True:
            success_flag = True

            # random image size
            img_size = (random.randint(256, 1024), random.randint(256, 1024))
            # img_size = (512, random.randint(256, 512+256))

            # Create a transparent image
            img = Image.new("RGBA", img_size, (255, 255, 255, 0))

            shapes_per_image = np.random.choice(num_shapes_per_image, p = num_shapes_per_image_prob)
            
            shape_alpha = 1.0

            annotations = {
                "image_size": img_size,
                "shape_alpha": shape_alpha,
                "num_shapes": shapes_per_image,
                "objects": []
            }

            if enforce_same_color:
                enforce_fill_color = enforce_outline_color = random_color(alpha=1.0)
            else:
                enforce_fill_color = enforce_outline_color = None
            
            if enforce_same_point_range:
                enforce_point_range = get_random_point_range(img_size[0], img_size[1])
            else:
                enforce_point_range = None

            if enforce_same_style:
                enforce_style = np.random.choice(shape_styles, p = shape_styles_prob)

            for j in range(shapes_per_image):
                tries = 50
                while True:
                    shape_type = np.random.choice(shape_types, p = shape_types_prob)
                    
                    if enforce_same_style:
                        style = enforce_style
                    else:
                        style = np.random.choice(shape_styles, p = shape_styles_prob)
                
                    if j > 0 and enforce_object_overlap:
                        # keep a copy of current image
                        prev_img = img.copy()

                    cur_obj_img, annotation = generate_shape(
                        img,
                        shape_type, 
                        img_size, 
                        style=style, 
                        alpha=shape_alpha,
                        enforce_fill_color=enforce_fill_color,
                        enforce_outline_color=enforce_outline_color,
                        enforce_point_range=enforce_point_range,
                        shape_range_configs=shape_range_configs,
                        select_shape_range_w_decreasing_prob=select_shape_range_w_decreasing_prob
                    )

                    if j == 0 or not enforce_object_overlap:
                        break
                    else:
                        # check if the filled map is a single connected component and does not entirely contain or be contained by the previous object
                        obj_count, _ = count_obj_num(img)
                        prev_binary_mask = np.where(np.array(prev_img)[:, :, 3] > 0, 1, 0)
                        cur_added_obj_binary_mask = np.where(np.array(cur_obj_img)[:, :, 3] > 0, 1, 0)
                        if_contained = do_masks_contain(prev_binary_mask, cur_added_obj_binary_mask)
                        # mask_iou = calculate_iou(prev_binary_mask, cur_added_obj_binary_mask)
                        intersection_ratio_prev, intersection_ratio_cur = intersection_ratio(prev_binary_mask, cur_added_obj_binary_mask)
                        # if obj_count == 1 and not if_contained and mask_iou < 0.8:
                        if obj_count == 1 and not if_contained and intersection_ratio_prev < 0.8 and intersection_ratio_cur < 0.8:
                            break
                        else:
                            # restore the previous image
                            img = prev_img
                    tries -= 1   
                    if tries == 0:
                        success_flag = False
                        break
                if not success_flag:
                    print("Failed to generate shape after max tries, try again...")
                    break
                else:
                    annotations['objects'].append(annotation)
            
            if success_flag and output_dir is not None:
                # data augmentation for robustness
                if data_augmentation is not None:
                    # set up noise color and ratio
                    if "random_noise" in data_augmentation:
                        object_color = annotations["objects"][0]["fill"] if annotations["objects"][0]["fill"] is not None else annotations["objects"][0]["outline"]
                        data_augmentation["random_noise"]["color"] = object_color
                        ratio_range = data_augmentation["random_noise"].get("ratio_range", (0.1, 0.5))
                        data_augmentation["random_noise"]["ratio"] = random.uniform(*ratio_range)
                        dilate_range = data_augmentation["random_noise"].get("dilate_range", (2, 5))
                        data_augmentation["random_noise"]["dilate_range"] = dilate_range
                        noise_range = data_augmentation["random_noise"].get("noise_range", (dilate_range[0], dilate_range[1]))
                        data_augmentation["random_noise"]["noise_size"] = random.randint(*noise_range)
                    img = img_augmentation_before_svg_conversion(img, data_augmentation)
                
                # Save the image with shapes
                img_out_path = os.path.join(output_img_dir, f"{output_prefix}{i}.png")
                img.save(img_out_path)

                if svg_config is not None:
                    svg_out_path = os.path.join(output_svg_dir, f"{output_prefix}{i}.svg")
                    svg_str = img_to_svg_str(img_out_path, svg_config, svg_out_path, truncate_len=None)

                if svg_str is None:
                    success_flag = False
                    print("Empty SVG, try again...")
                else:
                    if enforce_single_svg_path:
                        svg_path_num = check_svg_path_num(svg_str)
                        if svg_path_num > 1:
                            print("SVG path num > 1, try again...")
                            success_flag = False
                
                if not success_flag:
                    # remove the files
                    if os.path.exists(img_out_path):
                        os.remove(img_out_path)
                    if os.path.exists(svg_out_path):
                        os.remove(svg_out_path)
            

            if success_flag:
                if save_background:
                    # Save the background image by removing the shapes
                    background = Image.new("RGBA", img_size, (255, 255, 255, 0))
                    background_color = random_color(alpha=1)
                    img_rgba = img.convert("RGBA")
                    datas = img_rgba.getdata()
                    
                    newData = []
                    for item in datas:
                        if item[3] == 0:
                            newData.append(background_color)
                        else:
                            newData.append((255, 255, 255, 0))
                            
                    background.putdata(newData)
                    background_output_path = os.path.join(output_dir, f"{output_prefix}{i}_background.png")
                    background.save(background_output_path)
                        
            if success_flag:
                all_annotations['data'][f"{output_prefix}{i}"] = annotations
                if i % 1000 == 0:
                    # save partial annotations
                    with open(os.path.join(output_dir, f"annotations.json"), "w") as f:
                        json.dump(all_annotations, f, cls=NumpyEncoder)
                break

    print(len(all_annotations['data']))

    with open(os.path.join(output_dir, f"annotations.json"), "w") as f:
        json.dump(all_annotations, f, cls=NumpyEncoder)

### running main jobs
import argparse
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="single_obj, multi_obj")
    parser.add_argument("--split", type=str, required=True, default="train", help="train, eval")
    parser.add_argument("--multi_obj_style", type=str, required=False, default="outline", help="filled, outline")

    args = parser.parse_args()

    if args.split == "train":
        np.random.seed(42)
        random.seed(42)
    else:
        np.random.seed(0)
        random.seed(0)

    data_augmentation = {
        "random_noise": {"prob": 0.1, "ratio_range": (0.01, 0.05), "intensity_range": (0.1, 1.0), "dilate_range": (1, 3), "noise_size": (1, 3)},
        "blur": {"prob": 0.1, "radius": (0.1, 0.5)},
    }
    # data_augmentation = None

    if args.dataset == "single_obj":
        # run single object subset
        output_prefix = "single_obj_"
        styles = ["filled", "outline"]
        shape_styles_prob = [0.5, 0.5]
        svg_config = CONFIGS['pvd']
        print("svg config:", svg_config)
        
        ### v2.0 ###
        if args.split == "train":
            shape_types_and_num = [
                ("circle", 10000),
                ("ellipse", 10000),
                ("rectangle", 10000),
                ("triangle", 10000),
                ("polygon", 20000),
                ("line_segment", 10000),
                ("grid", 10000),
                ("path", 10000),
                ("graph", 10000)
            ]
        else:
            shape_types_and_num = [
                ("circle", 10),
                ("ellipse", 10),
                ("rectangle", 10),
                ("triangle", 10),
                ("polygon", 20),
                ("line_segment", 10),
                ("grid", 10),
                ("path", 10),
                ("graph", 10)
            ]
        ### ====== ###

        num_images = sum([num for _, num in shape_types_and_num])
        shape_types = [shape_type for shape_type, _ in shape_types_and_num]
        shape_type_prob = [num/num_images for _, num in shape_types_and_num]

        output_dir = f"../data/datasets/pvd_{args.split}/pvd-single_obj_{num_images//1000}k"
        print(f"Generating {num_images} images with 1 shape(s) per image")

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(shape_types_and_num, f)

        generate_dataset(
            num_images, 
            num_shapes_per_image = [1], 
            num_shapes_per_image_prob = None,
            shape_types = shape_types,
            shape_types_prob = shape_type_prob,
            shape_styles = styles,
            shape_styles_prob = shape_styles_prob,
            output_dir = output_dir,
            save_background = False,
            svg_config=svg_config,
            data_augmentation = data_augmentation,
            output_prefix=output_prefix
        )
    
    if args.dataset == "multi_obj":

        ## === v2.0 === ###
        if args.multi_obj_style == "filled":
            output_prefix = "multi_obj_filled_"
            styles = ["filled"]
            shape_styles_prob = [1.0]
            num_shapes_per_image = [2,3,4,5]
            num_shapes_per_image_prob = [0.4, 0.3, 0.2, 0.1]
        elif args.multi_obj_style == "outline":
            output_prefix = "multi_obj_outline_"
            styles = ["outline"]
            shape_styles_prob = [1.0]
            num_shapes_per_image = [2,3,4,5,6,7,8]
            num_shapes_per_image_prob = [0.3, 0.27, 0.23, 0.15, 0.025, 0.0125, 0.0125]

        if args.split == "train":
            if args.multi_obj_style == "filled":
                shape_types_and_num = [
                    ("circle", 5000),
                    ("rectangle", 5000),
                    ("triangle", 5000),
                    ("line_segment", 5000),
                ]
            else:
                shape_types_and_num = [
                    ("circle", 10000),
                    ("rectangle", 10000),
                    ("triangle", 10000),
                    ("line_segment", 10000),
                ]
        else:
            shape_types_and_num = [
                ("circle", 10),
                ("rectangle", 10),
                ("triangle", 10),
                ("line_segment", 10),
            ]
        shape_range_configs = DEFAULT_SHAPE_RANGE_CONFIGS

        ## ==================================== ###
        svg_config = CONFIGS['pvd']
        print("svg config:", svg_config)

        num_images = sum([num for _, num in shape_types_and_num])
        shape_types = [shape_type for shape_type, _ in shape_types_and_num]
        shape_type_prob = [num/num_images for _, num in shape_types_and_num]

        output_dir = f"../data/datasets/pvd_{args.split}/pvd-multi_obj_{args.multi_obj_style}_{num_images//1000}k"
        print(f"Generating {num_images} images with {num_shapes_per_image} shape(s) per image")

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(shape_types_and_num, f)

        generate_dataset(
            num_images, 
            num_shapes_per_image = num_shapes_per_image, 
            num_shapes_per_image_prob = num_shapes_per_image_prob,
            shape_types = shape_types,
            shape_types_prob = shape_type_prob,
            shape_styles = styles,
            shape_styles_prob = shape_styles_prob,
            shape_range_configs = shape_range_configs,
            output_dir = output_dir,
            save_background = False,
            svg_config=svg_config,
            data_augmentation = data_augmentation,
            output_prefix=output_prefix,
            enforce_object_overlap=True,
            enforce_same_color=True,
            enforce_same_style=True
        )

if __name__ == "__main__":
    main()