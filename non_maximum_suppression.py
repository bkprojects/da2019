import numpy as np


def suppress_non_maximum_patches(patch_array):
    """Return a list of local-maximum patches.
    IoU (Google: Intersection over Union) is used to sort out
    patches that are overlapping too much with better-scoring patches"""

    x1 = patch_array[:, 0]
    y1 = patch_array[:, 1]
    x2 = patch_array[:, 2]
    y2 = patch_array[:, 3]
    distances = patch_array[:, 4]
    # Check the math behind IoU for threshold: A(P1) = 100, A(P2) = 100,
    # Let area of intersection I(P1, P2) be 50 -> 50/(100+100-50) = 50 / 150 = 1/3 overlap
    threshold = 0.3
    areas = (x2 - x1) * (y2 - y1)
    # add [::-1] for highest values first if needed
    order = distances.argsort()

    maximum_patch_indices = []
    while order.size > 0:
        i = order[0]
        maximum_patch_indices.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        width_intersection = np.maximum(0.0, xx2 - xx1)
        height_intersection = np.maximum(0.0, yy2 - yy1)
        intersection = width_intersection * height_intersection
        intersection_over_union = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Get indices of patches where overlap calculated with IoU is < 50%
        indices_of_remaining_valid_patches = np.where(intersection_over_union <= threshold)[0]
        order = order[indices_of_remaining_valid_patches + 1]

    return maximum_patch_indices

