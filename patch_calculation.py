# coding=utf-8
import numpy as np
import scipy.spatial.distance as dist

""" Erzeugt einen Patch auf Grundlage der Übergebenene Variablen.

    Params:
        x1, y1: Punkt links oben des Patches
        x2, y2: Punkt rechts unten des Patches
        frames: Centroiden der zu analysierednen Pickledatei
        
    Returns:
        frames_with_mask: An den Patch angepasste Frames
        fullmask: Maske für den Patch
        
"""
def calc_patch(x1, y1, x2, y2, frames):

    np_frames = np.array(frames)

    # Maske erstellen, die den gesuchten momentanen Patch abdeckt

    # x/y Werte trennen
    x_values = np_frames[:,0]
    y_values = np_frames[:,1]

    # Maske für x/y Werte erstellen
    mask1 = (x1 <= x_values) & ( x_values <= x2)
    mask2 = (y1 <= y_values) & ( y_values <= y2)

    # Beide Masken kombinieren
    fullmask = [x & y for x,y in zip(mask1, mask2)]

    frames_with_mask = np_frames[fullmask]

    return fullmask, frames_with_mask


""" Vergleicht die Histogramme der Patches und sortiert diese

    Params:
        histogramm_list: Liste an Histogramme, die aus den Patches ausgewertet wurden
        compare_hist: Histogramm zu dem Anfragewort

    Returns:
        sort_dic: das sortierte Dictonary der Histogramme
            -> Key: Nummer zugehörig der histogramm_list
            -> Value: unterschied von hist_i zu compare_hist

"""
def sort_patches(histogramm_list, compare_hist):

    # shape anpassen -> damit hist_i - hist_compare gerechnet werden kann
    shaped_compare_hist = np.array([compare_hist, ] )
    # Für berechnung in numpy Objekte bringen
    histogramm_list_np = np.array(histogramm_list)


    # Distanz mit L2-Norm(euclidian)
    #distance = dist.cdist(histogramm_list_np, shaped_compare_hist, metric='euclidean')
    # Distanz mit cosine-similarity

    print('Distanzberechnung...')
    distance = dist.cdist(histogramm_list_np, shaped_compare_hist, metric='cosine')

    # zur auswertung nach der sortierung ein LuT
    hist__lut = {index: v_word for index, v_word in enumerate(distance, 0)}

    # Sortieren nach kleinstem wert -> am ähnlichsten an stelle 0
    sort_dic = sorted(hist__lut.items(), key=lambda kv: kv[1], reverse=False)

    return sort_dic



def calc_histogramms(desc_list,n_centroids):

    # Transformation desc_list (Descriptoren) in Histogramme
    histogramm_list = np.ones(n_centroids, dtype=int)

    for desc in desc_list:

        hist_temp = np.bincount(desc)

        # TODO: Warum kommt manchmal ein Histogramm mit n_centroids-1 raus ?
        # temporäre Lösung: Nullen füllen
        # Frage: ist das Histogramm dann verfälscht ?

        if len(hist_temp) != n_centroids:
            complete = n_centroids - len(hist_temp)
            hist_temp = np.insert(hist_temp, len(hist_temp), np.zeros(complete))

        histogramm_list = np.vstack((histogramm_list, hist_temp))



    histogramm_list = np.delete(histogramm_list, 0, 0)


    return histogramm_list



