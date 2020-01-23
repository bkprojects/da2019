# coding=utf-8
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

import seaborn as sns


def visualize_result(frames_list, desc_list, n_centroids, im_arr, cell_size, x_length, y_length,
                        best_patch_list, xy_coords_list_for_frames, best_patch_after_nms, maximum_patch_indices, x_step, y_step):

    # ---------------------------------------------------------
    # Visualisierung der besten Ergebnisse (Optional)
    # Erstellt einen Kasten um die besten Patches
    # frames_xy wird dann als optionales Argument an visualisation übergeben
    visualize = True
    frames_xy = []
    if visualize:
        show_ex = 100
        for i in range(0, show_ex):
            # 1. Beste Einträge aus dem Dict
            # 2. diesen Eintrag aus Frames entnehemen
            # 3. Tuple bilden um es einfacher in der visualisierung darzustellen:  [x,y] --> (x,y)
            frames_xy.append(tuple(best_patch_after_nms[i]))
    # ---------------------------------------------------------

    draw_centroids(frames_list, desc_list, n_centroids, im_arr, cell_size, frames_xy, x_length, y_length)

    # über alle patches
    create_heatmap(im_arr, best_patch_list, x_length, y_length, xy_coords_list_for_frames, x_step, y_step)

    # über nms-patches
    create_heatmap(im_arr, best_patch_list, x_length, y_length, xy_coords_list_for_frames, x_step, y_step, maximum_patch_indices)


def draw_centroids(frames, labels, n_centroids, im_arr, cell_size, sorted_histograms_value=None, x_lenght=None,
                   y_length=None):
    # Was soll gezeichten werden ?
    # Kasten um die Centroiden
    draw_descriptor_cells = False
    # Centroiden
    draw_centroid_points = False

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # optional: Kästen der besten Einträge zeichenen
    # color_i : zusätzliche sortierung nach Farben
    if not sorted_histograms_value is None:
        color_i = 0
        for x in sorted_histograms_value:
            if color_i < 3:
                rect = Rectangle(x, x_lenght, y_length, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
            elif color_i < 6:
                rect = Rectangle(x, x_lenght, y_length, linewidth=1, edgecolor='y', facecolor='none')
                ax.add_patch(rect)
            elif color_i < 15:
                rect = Rectangle(x, x_lenght, y_length, linewidth=1, edgecolor='magenta', facecolor='none')
                ax.add_patch(rect)
            elif color_i < 25:
                rect = Rectangle(x, x_lenght, y_length, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            else:
                rect = Rectangle(x, x_lenght, y_length, linewidth=1, edgecolor='grey', facecolor='none')
                ax.add_patch(rect)
            color_i += 1

    ax.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
    ax.autoscale(enable=False)
    colormap = cm.get_cmap('jet')
    desc_len = cell_size * 4

    if draw_centroid_points:
        for (x, y), label in zip(frames, labels):
            color = colormap(label / float(n_centroids))
            circle = Circle((x, y), radius=1, fc=color, ec=color, alpha=1)
            rect = Rectangle((x - desc_len / 2, y - desc_len / 2), desc_len, desc_len, alpha=0.08, lw=1)
            ax.add_patch(circle)

            if draw_descriptor_cells:
                for p_factor in [0.25, 0.5, 0.75]:
                    offset_dyn = desc_len * (0.5 - p_factor)
                    offset_stat = desc_len * 0.5
                    line_h = Line2D((x - offset_stat, x + offset_stat), (y - offset_dyn, y - offset_dyn), alpha=0.08,
                                    lw=1)
                    line_v = Line2D((x - offset_dyn, x - offset_dyn), (y - offset_stat, y + offset_stat), alpha=0.08,
                                    lw=1)
                    ax.add_line(line_h)
                    ax.add_line(line_v)
            ax.add_patch(rect)
    plt.show()


def mk_histogramm(labels, n_centroids):
    # Wie in Aufgabe 5 -> Berechne Histogramm für den Patch
    basis_lab = np.bincount(labels)
    colormap = cm.get_cmap('jet')
    bar_list = plt.bar(np.arange(len(basis_lab)), basis_lab)
    for label, bar in enumerate(bar_list):
        color = colormap(label / float(n_centroids))
        bar.set_color(color)
    plt.show()


def create_heatmap(im_arr, sorted_dict, x_l, y_l, xy_coords_list_for_frames, x_step, y_step,
                   maximum_patch_indices=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))

    inver_dict = {x: y for x, y in sorted_dict}

    sort_list = np.array(list(inver_dict.values()))

    iter_elements = range(len(sort_list))

    if not maximum_patch_indices is None:
        iter_elements = maximum_patch_indices

    heatmap = np.ones((3312, 2037))

    print('Anzahl Patches = %s' % len(sort_list))

    max_value = sort_list.max()
    min_value = sort_list.min()

    xy_coords_list_for_frames = np.array(xy_coords_list_for_frames)

    for i in iter_elements:
        for y in range(xy_coords_list_for_frames[i, 1], xy_coords_list_for_frames[i, 1] + y_l + 1):
            for x in range(xy_coords_list_for_frames[i, 0], xy_coords_list_for_frames[i, 0] + x_l + 1):
                if (not y >= 3310) and (not x >= 2035):
                    heatmap[y, x] += abs(sort_list[i] - max_value)


    plt.imshow(heatmap, cmap='jet', interpolation='sinc', alpha=0.7, vmin=heatmap.min(), vmax=heatmap.max())

    plt.show()

