from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def draw_text(input_image, content):
    """
    content is a dict. draws key: val on image
    Assumes key is str, val is float
    """
    image = input_image.copy()
    input_is_float = False
    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        image = (image * 255).astype(np.uint8)

    green = [57, 255, 20]
    margin = 15
    start_x = 5
    start_y = margin
    for key in sorted(content.keys()):
        value = content[key]
        if isinstance(value, str):
            text = '{}: {}'.format(key, value)
        else:
            text = "%s: %.2g" % (key, value)
        cv2.putText(image, text, (start_x, start_y), 0, 0.5, green, thickness=2)
        start_y += margin

    if input_is_float:
        image = image.astype(np.float32) / 255.
    return image


def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right heel
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left heel
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    19: left big toe
    20: right big toe
    21: Left small toe
    22: Right small toe
    23: L ankle
    24: R ankle
    """
    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink':  [197, 27, 125],  # L lower leg
        'light_pink': [233, 163, 201],  # L upper leg
        'light_green': [161, 215, 106],  # L lower arm
        'green': [77, 146, 33],  # L upper arm
        'red': [215, 48, 39],  # head
        'light_red': [252, 146, 114],  # head
        'light_orange': [252, 141, 89],  # chest
        'orange': [200,90,39],
        'purple': [118, 42, 131],  # R lower leg
        'light_purple': [175, 141, 195],  # R upper
        'light_blue': [145, 191, 219],  # R lower arm
        'blue': [69, 117, 180],  # R upper arm
        'gray': [130, 130, 130],  #
        'white': [255, 255, 255],  #
    }

    image = input_image.copy()
    input_is_float = False

    if (np.issubdtype(image.dtype, np.float32) or
            np.issubdtype(image.dtype, np.float64)):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white',
        'orange','light_orange','orange','light_orange','pink','light_pink'
    ]

    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])
        # Left is dark and right is light.
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 19:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    elif joints.shape[1] == 25:
        # parent indices -1 means no parents
        parents = np.array([
            24, 2, 8, 9, 3, 23, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15,
            16, 23, 24, 19, 20, 4, 1
        ])
        # Left is dark and right is light.
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue', # Right shoulder
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple',
            19: 'orange', # Left Big Toe
            20: 'light_orange', # Right Big Toe
            21: 'orange', # Left Small Toe
            22: 'light_orange', # Right Small Toe
            # Ankles!
            23: 'green', # Left
            24: 'gray'  # Right
        }
    else:
        print('Unknown skeleton!!')
        import ipdb
        ipdb.set_trace()

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'],
                       -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], -1)
        else:
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                import ipdb
                ipdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)
    return image
