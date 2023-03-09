IGNORE_LABEL=-1

LABELS = {
    -1: "unlabeled",
    0: "Plane",
    1: "Sphere",
    2: "Cylinder",
    3: "Cone",  
}

COLOR_MAP = {
    0:[0, 0, 0],
    1: [245, 150, 100],
    2: [245, 230, 100],
    3: [250, 80, 100],
    4: [150, 60, 30]
}



# objects which are not identifiable from a single scan are mapped to their closest
REMAPPING_MAP = {
    -1:-1,  # "unlabeled"
    0: 0, 
    1: 1,  
    2: 2, 
    3: 3,
}

# invert above feature map
LEARNING_MAP_INV = {
    -1:-1,  # "unlabeled"
    0: 0, 
    1: 1,  
    2: 2, 
    3: 3,
}

# sequences in split types
SPLIT = {"train": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17 ], "val": [18, 19], "test": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}

# sensor configuration
#SENSOR_CONFIG = {"name": "HDL64", "type": "spherical", "fov_up": 3, "fov_down": -25}

# # projected image properties
# IMG_PROP = {
#     # range, x, y, z signal
#     "img_means": [12.12, 10.88, 0.23, -1.04, 0.21],
#     "img_stds": [12.32, 11, 47, 6, 91, 0.86, 0.16],
# }

