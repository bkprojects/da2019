# coding=utf-8
import pickle as pickle
from collections import defaultdict
from itertools import combinations

import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt


from scipy.cluster.vq import kmeans2
from visualisation import draw_centroids
from visualisation import mk_histogramm
from visualisation import create_heatmap
from visualisation import visualize_result

from patch_calculation import calc_patch
from patch_calculation import sort_patches
from patch_calculation import calc_histogramms

from patch_coordinate_helper import create_array_for_nms

from non_maximum_suppression import suppress_non_maximum_patches

from spatial_pyramid_helper import calculate_spatial_pyramid_histograms
from spatial_pyramid_helper import calculate_spatial_pyramid_histogram

from calc_average_prec import calc_average_precision

from create_dic import createDictionary, getSelectedWordCoords


def patch_wordspotting():
    document_image_filename = '2700270.png'
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')

    # ------------------
    step_size = 15
    cell_size = 15
    # ------------------

    # Select Image and SIFT file
    pickle_densesift_fn = '2700270-full_dense-%d_sift-%d_descriptors.p' % (step_size, cell_size)
    frames, desc = pickle.load(open(pickle_densesift_fn, 'rb'))

    # ---------------
    n_centroids = 1024
    # ---------------

    # kmeans über dem gesamten dokument
    _, labels = kmeans2(desc, n_centroids, iter=20, minit='points')

    gtp_dict = createDictionary()

    visu = False
    if visu:
        word = 'the'
        # Vorkommen vom 'word' aus allem words
        index_from_word = 2
        coordinates = getSelectedWordCoords(gtp_dict, word, index_from_word)
        av, list_wit_params = find_similar_words(frames, n_centroids, labels, coordinates, gtp_dict[word])

        visualize_result(list_wit_params[0], list_wit_params[1], n_centroids, im_arr, cell_size, list_wit_params[5],
                         list_wit_params[6], list_wit_params[2], list_wit_params[3], list_wit_params[4], list_wit_params[7],
                         list_wit_params[8], list_wit_params[9])



    mean_average = calc_mean_average_precision(n_centroids, gtp_dict,  frames, labels)

    print('Mean Average: %s' % mean_average)



def calc_mean_average_precision(n_centroids, gtp_dict,  frames, labels):

    sum_of_averages = 0

    for key in gtp_dict:
        key_list = gtp_dict[key]
        for x1,y1,x2,y2 in key_list:
            print('Wort: %s' % key)
            print('Mit Koordinaten: %s,%s,%s,%s' % (x1, y1, x2, y2))
            average, _ = find_similar_words(frames, n_centroids, labels, (x1,y1,x2,y2), key_list)
            sum_of_averages += average

    mean_average = sum_of_averages / len(gtp_dict.values())

    return mean_average



def find_similar_words(frames, n_centroids, labels, coordinates,sample_coodinates):

    # Document size
    height = 3310
    width = 2034

    # Selected word: -- 580 319 723 406 the --
    word_x1, word_y1, word_x2, word_y2 = coordinates


    # x/y value fürs iterieren
    x = 0
    y = 0

    # -------------------
    # Patchgröße
    x_length = word_x2 - word_x1
    y_length = word_y2 - word_y1
    # -------------------


    # for testing
    #height = 1200

    # -----------------
    x_step = round(x_length/4)
    y_step = round(y_length/2)
    # -----------------


    # spatial parymid ? --------
    result_with_sp = True
    # --------------------------


    # Histogramm from Example word
    basic_desc_mask, _ = calc_patch(word_x1, word_y1, word_x2, word_y2, frames)
    compare_hist = np.bincount(labels[basic_desc_mask])

    if len(compare_hist) != n_centroids:
        complete = n_centroids - len(compare_hist)
        compare_hist = np.insert(compare_hist, len(compare_hist), np.zeros(complete))

    if result_with_sp:
        compare_hist = calculate_spatial_pyramid_histogram(compare_hist)

    xy_coords_list_for_frames = []

    # Iteration über das Dokument
    desc_list = []
    frames_list = []

    while y < height - y_length:
        while x < width - x_length:
            desc_mask,frames_patch = calc_patch(x, y, x + x_length, y + y_length, frames)
            frames_list.append(frames_patch)
            desc_list.append(labels[desc_mask])

            # x-Step -----------
            x = x + x_step
            # ------------------

            xy_coords_list_for_frames.append([x,y])

        # y-Step -----------
        x = 0
        y = y + y_step
        # ------------------
        # Fortschritt:
        print("%s %%" % round( y / height * 100 ))

    print('Patches berechnet')

    # Transformation desc_list (Descriptoren) in Histogramme
    histogram_list = calc_histogramms(desc_list, n_centroids)

    if result_with_sp:
        print('Berechne Spatial Pyramid')
        histogram_list = calculate_spatial_pyramid_histograms(histogram_list)


    # Patches sortieren nach ähnlichkeit zum Anfragewort
    # Dict mit ()
    # [(index, distance),..]
    best_patch_list = sort_patches(histogram_list, compare_hist)


    # Indizes von lokalen Maxima(beste Patches aus überlappendem Haufen) finden
    print('Berechne Non Maximum Suppression')
    nms_array = create_array_for_nms(best_patch_list, frames_list, x_length, y_length)
    maximum_patch_indices = suppress_non_maximum_patches(nms_array)

    # Berechnen der besten Patches als koordinaten
    best_patch_after_nms = np.array(xy_coords_list_for_frames)[ np.array(best_patch_list)[np.array(maximum_patch_indices)][:,0].astype(int)]


    average_presicion, recall = calc_average_precision(best_patch_after_nms, sample_coodinates, x_length, y_length)

    print('Recall: %s' % recall)
    print('Average Presicion: %s' % average_presicion)

    return average_presicion, [frames_list, desc_list,best_patch_list, xy_coords_list_for_frames,
                                        best_patch_after_nms, x_length, y_length, maximum_patch_indices, x_step, y_step]



if __name__ == '__main__':
    patch_wordspotting()
