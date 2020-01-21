import numpy as np
import logging

import pickle as pickle

import cv2
import argparse

def compute_sift_descriptors(im_arr, cell_size=5, step_size=20):
    # Generate dense grid
    frames = [(x, y) for x in np.arange(10, im_arr.shape[1], step_size)
              for y in np.arange(10, im_arr.shape[0], step_size)]

    # Note: In the standard SIFT detector algorithm, the size of the
    # descriptor cell size is related to the keypoint scale by the magnification factor.
    # Therefore the size of the descriptor is equal to cell_size/magnification_factor (
    # Default: 3)
    kp = [cv2.KeyPoint(x, y, cell_size / 3) for x, y in frames]

    sift = cv2.xfeatures2d.SIFT_create()

    sift_features = sift.compute(im_arr, kp)
    desc = sift_features[1]
    return frames, desc



if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)


    logger = logging.getLogger('SIFT::Compute')


    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', action='store', type=str,
                        default='2700270.png',
                        help='Path of page image')
    parser.add_argument('--cellsize', '-cs', action='store', type=int, default=5,
                        help='Size of a cell')
    parser.add_argument('--stepsize', '-ss', action='store', type=int, default=5,
                        help='Step size to define the dense grid')


    args = parser.parse_args()

    img = cv2.imread(args.file)

    logger.info('Computing SIFT descriptors for image %s. \n' +
                'Cell Size: %i, Step Size: %i',
                args.file, args.cellsize, args.stepsize)
    frames, desc = compute_sift_descriptors(img, args.cellsize, args.stepsize)
    logger.info('Computed  %i descriptors...', len(frames))

    pickle_densesift_fn = '2700270-full_dense-%d_sift-%d_descriptors.p' % (args.stepsize, args.cellsize)
    logger.info('Saving SIFT descriptors to %s', pickle_densesift_fn)
    pickle.dump((frames,desc), open(pickle_densesift_fn, 'wb'))


