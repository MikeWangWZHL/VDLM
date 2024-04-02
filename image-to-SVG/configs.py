DEFAULT_CONFIG = {
    "vtracer_config":{
        "colormode":'color',        # ["color"] or "binary"
        "hierarchical":'stacked',   # ["stacked"] or "cutout"
        "mode": "spline",            # ["spline"] "polygon", or "none"
        "filter_speckle": 4,         # default: 4
        "color_precision": 6,        # default: 6
        "layer_difference": 16,      # default: 16 "increase this on abstract image"
        "corner_threshold": 60,      # default: 60
        "length_threshold": 4,     # in [3.5, 10] default: 4.0
        "max_iterations": 10,        # default: 10
        "splice_threshold": 45,      # default: 45
        "path_precision": 8          # default: 8
    },
    "min_area_ratio": None, # min area = 16
    "resize_width": None, # resize image before translation
    "rescale_width": None, # resize SVG to a specific width
    "if_resize": True,
    "if_post_process": True,
    "simplify_threshold": None,
    "truncate_len": None
}

CONFIGS = {
    "pvd": {
        "vtracer_config":{
            "colormode":'color',        # ["color"] or "binary"
            "hierarchical":'stacked',   # ["stacked"] or "cutout"
            "mode": "polygon",            # ["spline"] "polygon", or "none"
            "filter_speckle": 4,         # default: 4
            "color_precision": 6,        # default: 6
            "layer_difference": 16,      # default: 16
            "corner_threshold": 60,      # default: 60
            "length_threshold": 4,     # in [3.5, 10] default: 4.0
            "max_iterations": 10,        # default: 10
            "splice_threshold": 45,      # default: 45
            "path_precision": 8          # default: 8
        },
        "min_area_ratio": 16,
        "resize_width": None,
        "if_resize": False,
        "rescale_width": None, # 512
        "if_post_process": True,
        "simplify_threshold": None,
        "truncate_len": 20480,
        "max_num_path": 1 # ensure that the number of paths is 1
    },
    "default": {
        "vtracer_config":{
            "colormode":'color',        # ["color"] or "binary"
            "hierarchical":'stacked',   # ["stacked"] or "cutout"
            "mode": "polygon",            # ["spline"] "polygon", or "none"
            "filter_speckle": 4,         # default: 4
            "color_precision": 6,        # default: 6
            "layer_difference": 16,      # default: 16
            "corner_threshold": 60,      # default: 60
            "length_threshold": 4,     # in [3.5, 10] default: 4.0
            "max_iterations": 10,        # default: 10
            "splice_threshold": 45,      # default: 45
            "path_precision": 8          # default: 8
        },
        "min_area_ratio": 16,
        "resize_width": None,
        "if_resize": False,
        "rescale_width": None, # 512
        "if_post_process": True,
        "simplify_threshold": None,
        "truncate_len": None,
        "max_num_path": None
    },
    "natural": {    
        "vtracer_config":{
            "colormode":'color',        # ["color"] or "binary"
            "hierarchical":'stacked',   # ["stacked"] or "cutout"
            "mode": "polygon",            # ["spline"] "polygon", or "none"
            "filter_speckle": 6,         # default: 4
            "color_precision": 6,        # default: 6
            "layer_difference": 64,      # default: 16
            "corner_threshold": 60,      # default: 60
            "length_threshold": 4,     # in [3.5, 10] default: 4.0
            "max_iterations": 10,        # default: 10
            "splice_threshold": 45,      # default: 45
            "path_precision": 8          # default: 8
        },
        "min_area_ratio": 16,
        "resize_width": 224,
        "rescale_width": 512,
        "if_resize": True,
        "if_post_process": True,
        "simplify_threshold": None,
        "truncate_len": None,
        "max_num_path": None
    },
}