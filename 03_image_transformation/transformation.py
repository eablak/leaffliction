import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUGMENTATION_DIR = PROJECT_ROOT / "02_data_augmentation"

sys.path.insert(0, str(AUGMENTATION_DIR))
os.chdir(AUGMENTATION_DIR)

from augmentation import get_directory, change_directory, handle_image


def arr_to_xy(point):
    arr = np.asarray(point)
    if arr.size == 0:
        return np.empty((0, 2), dtype=int)
    return arr.reshape(-1, 2)


def img_transformation(path):
    img = cv.imread(path)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_green = np.array([30, 40, 40])
    upper_green = np.array([85, 255, 255])

    g_blur = cv.GaussianBlur(img, (5, 5), 0)

    leaf_mask = cv.inRange(hsv, lower_green, upper_green)

    contours, _ = cv.findContours(leaf_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    overlay = img.copy()
    overlay[leaf_mask > 0] = (0, 255, 0)

    roi = img.copy()
    cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    pcv.params.sample_label = "plant"
    shape_image = pcv.analyze.size(img=img, labeled_mask=leaf_mask)

    # pcv.params.debug = "plot"
    # pcv.params.debug_outdir = "./outputs"

    pcv.params.sample_label = "plant"
    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(img=img, mask=leaf_mask)
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img, leaf_mask)

    plt.subplot(331), plt.imshow(img), plt.title("Original")
    plt.xticks([]), plt.yticks([])

    plt.subplot(332), plt.imshow(g_blur, cmap="gray"), plt.title("Gaussian blur")
    plt.xticks([]), plt.yticks([])

    plt.subplot(333), plt.imshow(leaf_mask), plt.title("Mask")
    plt.xticks([]), plt.yticks([])

    plt.subplot(334), plt.imshow(roi), plt.title("Roi objects")
    plt.xticks([]), plt.yticks([])

    plt.subplot(335), plt.imshow(shape_image), plt.title("Analyze object ")
    plt.xticks([]), plt.yticks([])

    plt.subplot(336), plt.imshow(img), plt.title("Pseudolandmarks")
    for point, color in [
        (left, "r"),
        (right, "b"),
        (center_h, "y"),
        (top, "g"),
        (bottom, "c"),
        (center_v, "m"),
    ]:
        point = arr_to_xy(point)
        plt.scatter(point[:, 0], point[:, 1], c=color, s=3)

    plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="Directory")
    parser.add_argument("-f", help="Image File")
    parser.add_argument("-flag", help="Flag for change directory name")

    args = parser.parse_args()
    directory = args.d
    img_file = args.f
    flag = args.flag

    if directory:

        # python3 transformation.py -d Apple -flag 1

        paths = get_directory(directory)
        print(paths)
        # transformation(directory, paths, "directory")
        if flag is not None and int(flag):
            change_directory()
        # save_images(img_file, transformation, "img_file")

    elif img_file:

        # python3 transformation.py -f "Apple_healthy/image (1).JPG"

        img_path = handle_image(img_file)
        print(img_path)
        img_transformation(img_path)

        # display_images(processed_images)
