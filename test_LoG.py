from scipy.ndimage import gaussian_laplace
from skimage.feature import peak_local_max
import numpy as np
from skimage.io import imsave, imread, imshow, MultiImage, ImageCollection
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import os
import time, os, sys
from urllib.parse import urlparse
# import matplotlib as mpl
# %matplotlib inline
# mpl.rcParams['figure.dpi'] = 300

from cellpose import models, utils, io
import glob
import fnmatch
import argparse


# @Artemiy: At this point, I just succeeded in making the script that locates the spots, and provides the intensity at their center.
# | I can't get my technique to produce spots as masks working since skimage "peak_local_max" only returns coordinates, instead of patches within a certain range.
# | It would imply to reimplement this function here, but I don't think that we will have the time.

def find_spots(imIn):
    # Denoising
    imMed = median_filter(imIn, 3)

    # Detecting spots and dying cells
    sigmas = [3, 5, 7]
    outs   = [gaussian_laplace(imMed, sigma) for sigma in sigmas]
    buffer = np.zeros(outs[0].shape, dtype=np.float32)

    for o in outs:
        buffer += np.divide(o, len(sigmas))

    buffer_norm = (buffer - np.min(buffer)) / (np.max(buffer) - np.min(buffer))
    coordinates = peak_local_max(buffer_norm, threshold_abs=0.4, min_distance=6)
    # print(len(coordinates))
    
    # imOut = np.zeros(buffer_norm.shape, dtype=np.float32)
    # for x, y in coordinates:
    #     imOut[x, y] = 1.0

    return coordinates



def attribute_spots(labeled_transmission, spots_coordinates, fluo_channel):
    
    # [label index] -> [intensity1, intensity2, ...]
    label_to_spots = {}

    for x, y in spots_coordinates:
        label = labeled_transmission[x, y]
        
        if label == 0:
            continue # We fell in the background
        
        # Adding intensity at the center of the spot.
        label_to_spots.setdefault(label, []).append(fluo_channel[x, y])

    return label_to_spots


# @Artemiy: You can remove this function and replace it by yours, just watch where it is used in the main().
def segment_transmission(transmission_channel, model):
    # Returns a labeled image (one value per individual)
    chan = [0,0]
    masks, flows, styles, diams = model.eval(transmission_channel, diameter=None, channels=chan)


    return masks


def main(input_dir):

    # Batching of an images folder:
    # @Artemiy: We maybe need some piece of GUI for the user to give a path ?
    content = os.listdir(input_dir)

    model = models.Cellpose(gpu=False, model_type='cyto')

    for c in content:
        if not c.lower().endswith("tif"):
            continue

        full_path = os.path.join(input_dir, c)

        if not os.path.isfile(full_path):
            continue
    
        print(f"Currently processing: {full_path}")

        image_col = ImageCollection(full_path)

        transmission_channel = np.array(image_col[0]) # @Artemiy: This is the transmission image. If you don't need it like that, you can remove this line.
        fluo_channel = np.array(image_col[1])

        spots_coordinates= find_spots(fluo_channel) # The result is a list of coordinates corresponding to points.
        labeled_transmission  = segment_transmission(transmission_channel, model) # @Artemiy: At this points, labeled_transmission must contain 1 label (==value) per individual.
        # This dictionary contains a list of intensity and is indexed by the value of labels.
        # This is not a list of list because provided labels are not necessarily contiguous.
        spots_intensity_lists = attribute_spots(labeled_transmission, spots_coordinates, fluo_channel)
        print(spots_intensity_lists)
    

parser = argparse.ArgumentParser()
args = parser.parse_args()
parser.add_argument(dest='input_dir', type=str, help="An input directory with TIFF images.", default="../yeast_test_data")
main("/home/artemiy/2023_defragmentation_school_porto/yeast_test_data")
# print(args.input_dir)

# @Artemiy: What do we make from the final data ?
# | Just a JSON ? stats over intensity (Q1, Q3, med & avg) + histogram of nb of spots/cell ?
# | If there are issues, we created a Discord yesterday, we can fix things after the dinner: https://discord.gg/wrYw8pSw