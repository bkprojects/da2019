import numpy as np
import warnings


def calc_average_precision(best_patch_after_nms, gtp_list, x_l, y_l):

    result_list = np.zeros(len(best_patch_after_nms))

    print('Anzahl der Vorkommen des Anfragewortes: %s' % len(gtp_list))

    warnings.filterwarnings("ignore")

    for x1, y1, x2, y2 in gtp_list:
        box2 = [x1, y1, x2, y2]
        for i in range(len(best_patch_after_nms)):
            temp_patch = best_patch_after_nms[i]
            box1 = [temp_patch[0], temp_patch[1], temp_patch[0] + x_l, temp_patch[1] + y_l]
            iou = calc_iou(box1, box2)
            if iou >= 0.5:
                result_list[i] = 1

    print('Anzahl gefundener Patches: %s' % np.sum(result_list))

    calc_av_mean = 0.
    for i in range(len(result_list)):
        try:
            calc_av_mean += (sum(result_list[:i+1]) / (i + 1)) * result_list[i]
        except:
            pass

    average_presicion = calc_av_mean / len(gtp_list)

    recall = np.sum(result_list) / len(gtp_list)

    return average_presicion, recall


def calc_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou





