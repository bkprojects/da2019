# coding=utf-8
import pickle as pickle
from itertools import combinations

import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt


from scipy.cluster.vq import kmeans2
from visualisation import draw_centroids
from visualisation import mk_histogramm
from patch_calculation import calc_patch
from patch_calculation import sort_patches
from patch_calculation import calc_histogramms
from patch_coordinate_helper import create_array_for_nms
from non_maximum_suppression import suppress_non_maximum_patches


def patch_wordspotting():
    document_image_filename = '2700270.png'
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
    #plt.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))

    # ------------------
    step_size = 15
    cell_size = 15
    # ------------------

    selectSIFT(step_size,cell_size, im_arr)
    plt.show()



def selectSIFT(step_size, cell_size, im_arr):

    # Select Image and SIFT file
    pickle_densesift_fn = '2700270-full_dense-%d_sift-%d_descriptors.p' % (step_size, cell_size)
    frames, desc = pickle.load(open(pickle_densesift_fn, 'rb'))

    # Document size
    height = 3310
    width = 2034

    # -------------------
    # Selected word: -- 580 319 723 406 the --
    word_x1 = 580
    word_y1 = 319
    word_x2 = 723
    word_y2 = 406

    # 436 330 567 434 for
    #word_x1 = 436
    #word_y1 = 330
    #word_x2 = 567
    #word_y2 = 434

    # 1263 1778 1430 1885 they
    #word_x1 = 1263
    #word_y1 = 1778
    #word_x2 = 1430
    #word_y2 = 1885

    # x/y value fürs iterieren
    x = 0
    y = 300

    # Patchgröße
    x_length = word_x2 - word_x1
    y_length = word_y2 - word_y1
    # -------------------


    # for testing
    height = 3310

    # -----------------
    x_step = round(x_length/6)
    y_step = round(y_length/4)
    # -----------------

    # ---------------
    n_centroids = 30
    # ---------------

    # kmeans über dem gesamten dokument
    _, labels = kmeans2(desc, n_centroids, iter=20, minit='points')


    # Histogramm from Example word
    basic_desc_mask, _ = calc_patch(word_x1, word_y1, word_x2, word_y2, frames)
    compare_hist = np.bincount(labels[basic_desc_mask])

    if len(compare_hist) != n_centroids:
        complete = n_centroids - len(compare_hist)
        compare_hist = np.insert(compare_hist, len(compare_hist), np.zeros(complete))


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

        # y-Step -----------
        x = 0
        y = y + y_step
        # ------------------
        # Fortschritt:
        print(y)


    # Transformation desc_list (Descriptoren) in Histogramme
    histogramm_list = calc_histogramms(desc_list, n_centroids)

    # Patches sortieren nach ähnlichkeit zum Anfragewort
    # Dict mit ()
    # [(index, distance),..]
    best_patch_dict = sort_patches(histogramm_list,compare_hist)

    # TODO: überlappende Patches aussortieren und besten Auswählen (done - bk)

    # TODO: cdist/cosine-> evaluation

    # TODO: Average Precision ermitteln

    # Indizes von lokalen Maxima(beste Patches aus überlappendem Haufen) finden
    nms_array = create_array_for_nms(best_patch_dict, frames_list, x_length, y_length)
    maximum_patch_indices = suppress_non_maximum_patches(nms_array)
    # ---------------------------------------------------------
    # Visualisierung der besten Ergebnisse (Optional)
    # Erstellt einen Kasten um die besten Patches
    # frames_xy wird dann als optionales Argument an visualisation übergeben
    visualize = True
    frames_xy = []
    if visualize:
        show_ex = 15
        for i in range(0,show_ex):
            # 1. Beste Einträge aus dem Dict
            # 2. diesen Eintrag aus Frames entnehemen
            # 3. Tuple bilden um es einfacher in der visualisierung darzustellen:  [x,y] --> (x,y)
            frames_xy.append(tuple(frames_list[best_patch_dict[maximum_patch_indices[i]][0]][0]))
    print(frames_xy)
    # ---------------------------------------------------------

    draw_centroids(frames_list,desc_list,n_centroids,im_arr,cell_size,frames_xy,x_length,y_length)



if __name__ == '__main__':
    patch_wordspotting()
