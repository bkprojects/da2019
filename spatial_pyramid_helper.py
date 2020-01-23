import numpy as np


def append_histograms(row):
    arr1, arr2 = np.array_split(row, 2)
    return np.concatenate((row, arr1, arr2))


def calculate_spatial_pyramid_histogram(histogram):
    arr1, arr2 = np.array_split(histogram, 2)
    extended_histogram = np.concatenate((histogram, arr1, arr2))
    #print(extended_histogram)
    return extended_histogram


def calculate_spatial_pyramid_histograms(histogram_array):
    extended_histogram = np.apply_along_axis(append_histograms, 1, histogram_array)
    return extended_histogram
