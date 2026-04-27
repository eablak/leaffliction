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

from augmentation import handle_image


def get_directory(dir_name):

    path = Path(os.getcwd()).parent / "augmented_directory" / "images"

    paths = {}
    for x in os.listdir(path):
        if dir_name in x:
            paths[x] = str(path / x)

    return list(paths.values())[0]


def batch_maker(path, batch_size):
    files = [
        os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".jpg"))
    ]

    for i in range(0, len(files), batch_size):
        yield files[i : i + batch_size]


def save_image(img, path):
    cv.imwrite(path, img)


def arr_to_xy(point):
    arr = np.asarray(point)
    if arr.size == 0:
        return np.empty((0, 2), dtype=int)
    return arr.reshape(-1, 2)


def apply_g_blur(img):
    return cv.GaussianBlur(img, (5, 5), 0)


def aplly_mask(hsv, l, u):
    return cv.inRange(hsv, l, u)


def apply_roi(mask, img):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    overlay = img.copy()
    overlay[mask > 0] = (0, 255, 0)

    roi = img.copy()
    return cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)


def apply_anlyze(img, leaf_mask):
    pcv.params.sample_label = "plant"
    return pcv.analyze.size(img=img, labeled_mask=leaf_mask)


def apply_pseu(img, leaf_mask, save=None):
    pcv.params.sample_label = "plant"

    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(img=img, mask=leaf_mask)
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(img, leaf_mask)

    if save:
        pseu_img = img.copy()

        points = [
            (left, "r"),
            (right, "b"),
            (top, "g"),
            (bottom, "c"),
        ]

        for pts, color in points:
            pts = arr_to_xy(pts)

            for x, y in pts:
                cv.circle(pseu_img, (int(x), int(y)), 2, (0, 0, 255), -1)

        return pseu_img
    else:
        return left, right, center_h, top, bottom, center_v


def img_transformation(path, process=None):
    img = cv.imread(path)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_green = np.array([30, 40, 40])
    upper_green = np.array([85, 255, 255])

    if process == None:
        g_blur = apply_g_blur(img)

        leaf_mask = aplly_mask(hsv, lower_green, upper_green)

        roi = apply_roi(leaf_mask, img)
        shape_image = apply_anlyze(img, leaf_mask)
        left, right, center_h, top, bottom, center_v = apply_pseu(img, leaf_mask)

    else:

        if process == "blur":
            return apply_g_blur(img)

        elif process == "mask":
            return aplly_mask(hsv, lower_green, upper_green)

        elif process == "roi":
            leaf_mask = aplly_mask(hsv, lower_green, upper_green)
            return apply_roi(leaf_mask, img)

        elif process == "analyze":
            leaf_mask = aplly_mask(hsv, lower_green, upper_green)
            return apply_anlyze(img, leaf_mask)
        else:
            leaf_mask = aplly_mask(hsv, lower_green, upper_green)
            return apply_pseu(img, leaf_mask, True)

    return (
        img,
        g_blur,
        leaf_mask,
        roi,
        shape_image,
        left,
        right,
        center_h,
        top,
        bottom,
        center_v,
    )


def display_images(
    img,
    g_blur,
    leaf_mask,
    roi,
    shape_image,
    left,
    right,
    center_h,
    top,
    bottom,
    center_v,
):
    plt.subplot(331), plt.imshow(img), plt.title("Original")
    plt.xticks([]), plt.yticks([])

    plt.subplot(332), plt.imshow(g_blur, cmap="gray"), plt.title("Gaussian blur")
    plt.xticks([]), plt.yticks([])

    plt.subplot(333), plt.imshow(leaf_mask, cmap="gray"), plt.title("Mask")
    plt.xticks([]), plt.yticks([])

    plt.subplot(334), plt.imshow(roi), plt.title("Roi objects")
    plt.xticks([]), plt.yticks([])

    plt.subplot(335), plt.imshow(shape_image), plt.title("Analyze object ")
    plt.xticks([]), plt.yticks([])

    plt.subplot(336), plt.imshow(img), plt.title("Pseudolandmarks")
    for point, color in [
        (left, "r"),
        (right, "b"),
        (top, "g"),
        (bottom, "c"),
    ]:
        point = arr_to_xy(point)
        plt.scatter(point[:, 0], point[:, 1], c=color, s=3)
    plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", help="Source")
    parser.add_argument("-f", help="Image File")
    parser.add_argument("-d", help="Destination")
    parser.add_argument("-p", help="process", default=None)

    args = parser.parse_args()
    source = args.s
    img_file = args.f
    dest = args.d
    process = args.p

    all_process = ["blur", "mask", "roi", "analyze", "pseu"]
    if process != None:
        if process not in all_process:
            raise ValueError("The process is not available")

    if source and dest:

        # python .\03_image_transformation\transformation.py -s Apple_Black_rot -d masked -p mask
        dest_dir = Path.cwd().parent / dest / source
        dest_dir.mkdir(parents=True, exist_ok=True)
        source_path = get_directory(source)
        for batch in batch_maker(source_path, batch_size=4):
            for img_path in batch:
                img = img_transformation(img_path, process)
                if img is None or isinstance(img, bool):
                    continue
                filename = os.path.basename(img_path)
                name, ext = os.path.splitext(filename)

                save_path = os.path.join(dest_dir, f"{name}_{process}{ext}")

                save_image(img, save_path)

    elif img_file:

        # python3 transformation.py -f "Apple_healthy/image (1).JPG"

        img_path = handle_image(img_file)
        print(img_path)
        (
            img,
            g_blur,
            leaf_mask,
            roi,
            shape_image,
            left,
            right,
            center_h,
            top,
            bottom,
            center_v,
        ) = img_transformation(img_path)

        display_images(
            img,
            g_blur,
            leaf_mask,
            roi,
            shape_image,
            left,
            right,
            center_h,
            top,
            bottom,
            center_v,
        )

    else:
        raise ValueError("Something went wrong")
