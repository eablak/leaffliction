import argparse
import os
from pathlib import Path
import sys
import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
import matplotlib.pyplot as plt
from deskew import determine_skew
import matplotlib
from wand.image import Image
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from block_distortion import distort_image


def handle_image(file_path):
    
    path = os.getcwd()
    path += "/../leaves/images/" + file_path
    path = Path(path)

    if path.exists():
        return path
    
    sys.exit("File not exists!")


def img_processing(img_path):

    src = cv2.imread(img_path)
    rows, cols, ch = src.shape

    # Flip
    flipped_image = cv2.flip(src, 0)

    # Rotate
    rotated_image = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)

    # Skew
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(src, M, (cols, rows))

    # Shear
    M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    sheared_img = cv2.warpPerspective(src, M, (int(cols*1.5), int(rows*1.5)))

    # Crop
    cropped_img = src[50:280, 50:300]

    # Distortion
    distorted_img = distort_image(src)


    # Save processed images instead of displaying
    cv2.imwrite("flipped_image.jpg", flipped_image)
    cv2.imwrite("rotated_image.jpg", rotated_image)
    cv2.imwrite("skewed_image.jpg", dst)
    cv2.imwrite("sheared_image.jpg", sheared_img)
    cv2.imwrite("cropped_image.jpg", cropped_img)
    imsave("distorted_image.jpg", img_as_ubyte(distorted_img))
    print("Images saved successfully!")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help="Directory")
    parser.add_argument('-f', help="Image File")

    args = parser.parse_args()
    directory = args.d
    img_file = args.f

    if directory:
        print("directoryy")
    elif img_file:
        # python3 augmentation.py -f "Apple_healty/image (1).JPG"
        img_path = handle_image(img_file)
        img_processing(img_path)