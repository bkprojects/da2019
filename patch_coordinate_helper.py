import numpy as np

def calculate_bottom_right_coords(x_left, y_top, x_length, y_length):
    x_right = x_left + x_length
    y_bottom = y_top + y_length
    return x_right, y_bottom

def create_array_for_nms(ranked_patches, frames_list, x_length, y_length):
    """Returns an array with 5 columns containing x1,y1,x2,y2 and distance-value
       for every patch which is then used for the NMS-Algorithm"""
    nms_array = []
    for patch in ranked_patches:
        index = patch[0]
        x_left, y_top = frames_list[index][0]
        distance_value = patch[1]
        print(distance_value)
        x_right, y_bottom = calculate_bottom_right_coords(x_left, y_top, x_length, y_length)
        nms_array = np.append(nms_array, [x_left, y_top, x_right, y_bottom, distance_value])

    nms_array = np.array(nms_array).reshape(-1, 5)
    return nms_array
