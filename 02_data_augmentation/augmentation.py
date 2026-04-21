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
matplotlib.use("TkAgg")
from wand.image import Image
from skimage import img_as_ubyte
from skimage.io import imread, imsave
from block_distortion import distort_image
from PIL import Image


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

    return [flipped_image, rotated_image, dst, sheared_img, cropped_img, distorted_img]


def save_images(file_name, processed_images, process):

    if process == "img_file":

        x = file_name.split("/")
        img_folder = x[0]
        img_number = x[1].replace(".JPG", "")

        path = os.getcwd()
        path += "/../leaves/images/" + img_folder + "/"
        path = Path(path)
    
    else:
        path = file_name
        img_number = path.split("/")[-1].replace(".JPG", "")
        path = path.rsplit("/", 1)[0]
        path = Path(path)

    if path.exists():
        cv2.imwrite(os.path.join(path, img_number + "_Flip.JPG"), processed_images[0])
        cv2.imwrite(os.path.join(path, img_number + "_Rotate.JPG"), processed_images[1])
        cv2.imwrite(os.path.join(path, img_number + "_Skew.JPG"), processed_images[2])
        cv2.imwrite(os.path.join(path, img_number + "_Shear.JPG"), processed_images[3])
        cv2.imwrite(os.path.join(path, img_number + "_Crop.JPG"), processed_images[4])
        imsave(os.path.join(path, img_number + "_Distortion.JPG"), img_as_ubyte(processed_images[5]))
    

def display_images(processed_images):

    fig = plt.figure(figsize=(10, 7))

    titles = ["Flip", "Rotate", "Skew", "Shear", "Crop", "Distortion"]

    for i, img in enumerate(processed_images):
        ax = plt.subplot(2, 3, i+1)

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img)
        ax.axis('off')
        plt.title(titles[i])

    plt.show()


def get_directory(dir_name):
    
    path = os.getcwd()
    path += "/../leaves/images/"
    path = Path(path)

    paths = {}
    for x in os.listdir(path):
        if dir_name in x:
            paths[x] = (str(path) + "/" + x)

    return paths


def per_augmentation(target_path, paths, process, stop_count):

    files = os.listdir(paths[target_path])
    static_files = []
    augmentations =  ["_Flip", "_Rotate", "_Skew", "_Shear", "_Crop", "_Distortion"]

    for file in files:
        if not any(aug in file for aug in augmentations):
            static_files.append(file)

    for i in range(len(static_files)):
        
        img_path = paths[target_path] + "/" + static_files[i]
        processed_images = img_processing(img_path)
        save_images(img_path, processed_images, process)
        if i == stop_count:
            break
    
    print(f"{target_path} augmentation successful!")


def augmentation(directory, paths, process):

    if directory == "Apple":
        
        per_augmentation("Apple_rust", paths, process, 220)
        per_augmentation("Apple_Black_rot", paths, process, 160)
        per_augmentation("Apple_scab", paths, process, 170)

    if directory == "Grape":

        per_augmentation("Grape_healthy", paths, process, 160)
        per_augmentation("Grape_Black_rot", paths, process, 30)
        per_augmentation("Grape_spot", paths, process, 50)


def change_directory():

    path = os.getcwd()
    path += "/../"

    for x in os.listdir(path):
        if x == "leaves":
            os.rename(path+x,path+"augmented_directory")
    
    print("Directory name changed!")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help="Directory")
    parser.add_argument('-f', help="Image File")
    parser.add_argument('-flag', help="Flag for change directory name")

    args = parser.parse_args()
    directory = args.d
    img_file = args.f
    flag = args.flag

    if directory:

        # python3 augmentation.py -d Apple -flag 1

        paths = get_directory(directory)
        augmentation(directory, paths, "directory")
        if flag is not None and int(flag):
            change_directory()

    elif img_file:
        
        # python3 augmentation.py -f "Apple_healthy/image (1).JPG"
        
        img_path = handle_image(img_file)
        processed_images = img_processing(img_path)
        
        save_images(img_file, processed_images, "img_file")
        display_images(processed_images)