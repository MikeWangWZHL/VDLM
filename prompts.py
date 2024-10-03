
DEFAULT_REASONING_PROMPT = '''The following JSON contains an approximated perception of a 2d scene. Each object (potentially including the background) is represented by a list of geometric shapes. If the object contain multiple shapes, it is a composite object. The (x, y) coordinates are in pixels, and (0, 0) is the top-left corner of the image.

--- perception ---
{perception}
------

Note that perception can be noisy. Make educated guesses if necessary. 
Think step by step and answer the following question:
{question}
'''



### pvd -> downstream tasks prompt ###

dt_pvd__length_comparison = '''The following JSON contains an approximated perception of a 2d scene. Each object (potentially including the background) is represented by a list of geometric shapes. If the object contain multiple shapes, it is a composite object. The (x, y) coordinates are in pixels, and (0, 0) is the top-left corner of the image.

--- perception ---
{perception}
------

Answer the following question:
Are these two lines of equal length?

Note that perception can be noisy. A 5% offset in the measurements is acceptable. You MUST provide your final answer, and the answer should follow this format: {{"answer": "yes" or "no"}}.
'''

dt_pvd__acute_or_obtuse = '''The following JSON contains an approximated perception of a 2d scene. Each object (potentially including the background) is represented by a list of geometric shapes. If the object contain multiple shapes, it is a composite object. The (x, y) coordinates are in pixels, and (0, 0) is the top-left corner of the image.

--- perception ---
{perception}
------

Answer the following question:
Is this an acute angle or an obtuse angle? 

Note that if the perception result includes a path, the angle is defined with the midpoint as the vertex, and the rays extend from the midpoint to both the head and end points. For example, if a path from A to B to C is perceived, the angle is defined as the angle between the vector BA and BC.
You MUST provide your final answer, and the answer should follow this format: {{"answer": "acute" or "obtuse"}}.
'''

dt_pvd__shapeworld_spatial_twoobj = '''Given a 2d scene containing objects with the following attributes:
--- Attribute Ontology ---
- color: ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray']
- shape: ['square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse']
------

The following JSON contains an approximated perception of the scene. Each object (potentially including the background) is represented by a list of geometric shapes. If the object contain multiple shapes, it is a composite object. The (x, y) coordinates are in pixels, and (0, 0) is the top-left corner of the image.

--- perception ---
{perception}
------

Note that the perception can be noisy. First identify the best matching shape type and the color type from the ontology for each perceived object. For composite objects, please match the entire composition to one of the most probable objects in the ontology. Make educated guesses if necessary. Then, think step by step and answer the following question:
{question}

You MUST provide your final answer, and the answer should follow this format: {{"answer": "yes" or "no"}}
'''

dt_pvd__shapeworld_superlative = '''Given a 2d scene containing objects with the following attributes:
--- Attribute Ontology ---
- color: ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray']
- shape: ['square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse']
------

The following JSON contains an approximated perception of the scene. Each object (potentially including the background) is represented by a list of geometric shapes. If the object contain multiple shapes, it is a composite object. The (x, y) coordinates are in pixels, and (0, 0) is the top-left corner of the image. The lowermost object has the largest y-coordinate, and the rightmost object has the largest x-coordinate.

--- perception ---
{perception}
------

Note that the perception can be noisy. First identify the best matching shape type and the color type from the ontology for each perceived object. For composite objects, please match the entire composition to one of the most probable objects in the ontology. Make educated guesses if necessary. Then, think step by step and answer the following question:
{question}

You MUST provide your final answer, and the answer should follow this format: {{"answer": "yes" or "no"}}
'''

dt_pvd__shapeworld_spatial_multiobj = '''Given a 2d scene containing objects with the following attributes:
--- Attribute Ontology ---
- color: ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray']
- shape: ['square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse']
------

The following JSON contains an approximated perception of the scene. Each object (potentially including the background) is represented by a list of geometric shapes. If the object contain multiple shapes, it is a composite object. The (x, y) coordinates are in pixels, and (0, 0) is the top-left corner of the image. If two objects overlap, the one with the larger index is considered to be in front of the other.

--- perception ---
{perception}
------

Note that the perception can be noisy. First identify the best matching shape type and the color type from the ontology for each perceived object. For composite objects, please match the entire composition to one of the most probable objects in the ontology. Make educated guesses if necessary. Then, think step by step and answer the following question:
{question}

You MUST provide your final answer, and the answer should follow this format: {{"answer": "yes" or "no"}}
'''

dt_pvd__maze = '''The following JSON contains an approximated perception of a {n}x{n} maze. Each object (potentially including the background) is represented by a list of geometric shapes. If the object contains multiple shapes, it is a composite object. The (x, y) coordinates for the vertices and edges correspond to the width and height position in pixels, and (0, 0) is the top-left corner of the image.

--- perception ---
{perception}
------

In the {n}x{n} maze, walls are depicted by a grid of black lines between cells and along the perimeter. The maze cells are defined within the grid. The start position is marked by a red circle, and the end position by a red star. The position of each cell can be represented by coordinates (row, column), beginning from the top-left corner as (0, 0). Here, 'row' corresponds to the vertical dimension (height) of the image, and 'column' to the horizontal dimension (width).

Perform the following steps to solve the maze:
(1) Infer the connectivity of the cells using a connection list. For example, a {n}x{n} maze should have a `connection_list` containing two sublisits with dimension {m}x{n} and {n}x{m}. For i in range(0, {m}) and j in range(0, {n}), `connection_list[0][i][j]` is `True` if cell `(i, j)` is vertically connected to cell `(i+1, j)` without being seperated by a wall. Similarly, for j in range(0, {m}) and i in range(0, {n}), `connection_list[1][i][j]` is `True` if cell `(i, j)` is horizontally connected to cell `(i, j+1)` without being seperated by a wall.
(2) Infer the start position and end position of the maze in the row-column format.
(3) Solve the maze by finding a path from the start position to the end position.
You MUST provide your final answer, and the answer should follow this format: {{"solution": "a list of (row, column) coordinates"}}.
'''

dt_pvd__nlvr = '''Given an image containing three boxes with light grey background, horizontally laid out. The boxes are separated by two dark grey rectangles placed vertically, which are referred to as walls. An "edge" is referred to as the boundary of the image. A "base edge" is referred to as the bottom boundary. If an object's boundary is located very close to an edge or a wall (e.g., within 10 pixels), it is considered as "touching". Each box contains a set of shapes. There are two types of images: "Tower", and "Scatter". In "Tower" images, each box contains only squares stacked in towers with up to 4 squares. In "Scatter" images, each box contains scattered objects of different sizes and shapes.

The following JSON contains an approximated perception of the image. Each object (potentially including the background) is represented by a list of geometric shapes. If the object contain multiple shapes, it is a composite object. The (x, y) coordinates are in pixels, and (0, 0) is the top-left corner of the image.

--- perception ---
{perception}
------

Now, identify the content in each box based on the perception result, and then think step by step to answer the following question:
{question}

You MUST provide your final answer, and the answer should follow this format: {{"answer": "yes" or "no"}}
'''

dt_pvd__geoclidean = '''The following JSON contains an approximated perception of the image. Each object (potentially including the background) is represented by a list of geometric shapes. If the object contain multiple shapes, it is a composite object. The (x, y) coordinates are in pixels, and (0, 0) is the top-left corner of the image.

--- perception ---
{perception}
------

The top part of the scene provides {n_shot} reference examples of a Euclidean geometry concept. A Euclidean geometry concept consists of a composition of one or more primitive geometric shapes, such as circles and line segments, with some constraints. These constraints include but are not limited to lengths, angles, and spatial relationships between the primitive shapes, for example, two perpendicular line segments or an equilateral triangle, etc. 
The bottom part of the scene presents a test example, separated from the top part by a red horizontal line. 
First, identify the pattern and constraints of the reference and test concepts based on the perception result. Note that the perception can be noisy. Make educated guesses if necessary. 
Then, determine if the test example depicts the same concept as the reference examples. 

You MUST provide your final answer, and the answer should follow this format: {{"answer": "yes" or "no"}}.
'''

dt_pvd__vgbench_qa = '''
Given an image containing a SVG graphic, think step by step and answer the following question:
{question}

{options}

------
The following JSON contains an approximated reference perception of the image. Each object (potentially including the background) is represented by a list of geometric shapes. If the object contain multiple shapes, it is a composite object. The (x, y) coordinates are in pixels, and (0, 0) is the top-left corner of the image.

--- reference perception ---
{perception}
------

Note that the reference perception can be noisy. Refer to the reference perception when necessary for answering the question.

You MUST provide your final answer, and the answer should follow this format: {{"answer": choose from "A", "B", "C", "D"}}
'''

downstream_task_pvd_prompts = {
    "length-comparison": dt_pvd__length_comparison,
    "acute-or-obtuse": dt_pvd__acute_or_obtuse,
    "shapeworld-spatial-2obj": dt_pvd__shapeworld_spatial_twoobj,
    "shapeworld-spatial-multiobj": dt_pvd__shapeworld_spatial_multiobj,
    "shapeworld-superlative": dt_pvd__shapeworld_superlative,
    "maze": dt_pvd__maze,
    "nlvr": dt_pvd__nlvr,
    "geoclidean": dt_pvd__geoclidean,
    "vgbench_qa_svg_category": dt_pvd__vgbench_qa,
    "vgbench_qa_svg_color": dt_pvd__vgbench_qa,
    "vgbench_qa_svg_usage": dt_pvd__vgbench_qa
}


### downstream task - image input propmts

dt_img__length_comparison = '''Are these two lines of equal length? Choose "Yes" or "No".
'''

dt_img__acute_or_obtuse = '''Is this an acute angle or an obtuse angle? Answer with "Acute" or "Obtuse".
'''

dt_img__shapeworld = '''Given a 2d scene containing objects with the following attributes:
--- Attribute Ontology ---
- color: ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray']
- shape: ['square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse']
------

Think step by step and answer the following question:
{question}

You MUST provide your final answer, and the answer should follow this format: {{"answer": "yes" or "no"}}
'''

dt_img__maze = '''Given a {n}x{n} maze, where the walls are depicted by a grid of black lines between cells and along the perimeter. The maze cells are defined within the grid. The start position is marked by a red circle, and the end position by a red star. The position of each cell can be represented by coordinates (row, column), beginning from the top-left corner as (0, 0). Here, 'row' corresponds to the vertical dimension (height) of the image, and 'column' to the horizontal dimension (width).

Solve the maze by finding a path from the start position to the end position.
You MUST provide your final answer, and the answer should follow this format: {{"solution": "a list of (row, column) coordinates"}}.
'''

dt_img__nlvr = '''Given an image containing three boxes with light grey background, horizontally layed out. The boxes are seperated by two dark grey rectangles placed vertically, which are referred to as walls. An "edge" is referred to as the boundary of the image. A "base edge" is referred to as the bottom boundary. If an object's boundary is located very close to an edge or a wall (e.g., within 10 pixels), it is considered as "touching". Each box contains a set of shapes. There are two types of images: "Tower", and "Scatter". In "Tower" images, each box contains only squares stacked in towers with up to 4 squares. In "Scatter" images, each box contains scattered objects of different sizes and shapes.

Think step by step to answer the following question:
{question}

You MUST provide your final answer, and the answer should follow this format: {{"answer": "yes" or "no"}}
'''

dt_img__geoclidean = '''The top part of the scene provides {n_shot} reference examples of a Euclidean geometry concept. A Euclidean geometry concept consists of a composition of one or more primitive geometric shapes, such as circles and line segments, with some constraints. These constraints include but are not limited to lengths, angles, and spatial relationships between the primitive shapes, for example, two perpendicular line segments or an equilateral triangle, etc. 
The bottom part of the scene presents a test example, separated from the top part by a red horizontal line. 
First, identify the pattern and constraints of the reference and test concepts. 
Then, determine if the test example depicts the same concept as the reference examples. 

You MUST provide your final answer, and the answer should follow this format: {{"answer": "yes" or "no"}}.
'''

dt_img__vgbench_qa = '''Given an image containing a SVG graphic, think step by step and answer the following question:
{question}

{options}

You MUST provide your final answer, and the answer should follow this format: {{"answer": choose from "A", "B", "C", "D"}}
'''

downstream_task_image_prompts = {
    "length-comparison": dt_img__length_comparison,
    "acute-or-obtuse": dt_img__acute_or_obtuse,
    "shapeworld-spatial-2obj": dt_img__shapeworld,
    "shapeworld-spatial-multiobj": dt_img__shapeworld,
    "shapeworld-superlative": dt_img__shapeworld,
    "maze": dt_img__maze,
    "nlvr": dt_img__nlvr,
    "geoclidean": dt_img__geoclidean,
    "vgbench_qa_svg_category": dt_img__vgbench_qa,
    "vgbench_qa_svg_color": dt_img__vgbench_qa,
    "vgbench_qa_svg_usage": dt_img__vgbench_qa
}


