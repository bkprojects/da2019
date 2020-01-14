# coding=utf-8
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


def draw_centroids(frames,labels,n_centroids,im_arr,cell_size,sorted_histograms_value=None,x_lenght=None,y_length=None):

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
            else:
                rect = Rectangle(x, x_lenght, y_length, linewidth=1, edgecolor='r', facecolor='none')
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
                    line_h = Line2D((x - offset_stat, x + offset_stat), (y - offset_dyn, y - offset_dyn), alpha=0.08, lw=1)
                    line_v = Line2D((x - offset_dyn, x - offset_dyn), (y - offset_stat, y + offset_stat), alpha=0.08, lw=1)
                    ax.add_line(line_h)
                    ax.add_line(line_v)
            ax.add_patch(rect)
    plt.show()




def mk_histogramm(labels,n_centroids):
    # Wie in Aufgabe 5 -> Berechne Histogramm für den Patch
    basis_lab = np.bincount(labels)
    colormap = cm.get_cmap('jet')
    bar_list = plt.bar(np.arange(len(basis_lab)), basis_lab)
    for label, bar in enumerate(bar_list):
        color = colormap(label / float(n_centroids))
        bar.set_color(color)
    plt.show()
