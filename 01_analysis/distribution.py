import argparse
from pathlib import Path
import os
import sys
import re
import matplotlib.pyplot as plt


def get_images(directory):

    if directory == "Apple" or directory == "Grape":

        path = os.getcwd()
        path += "/../leaves/images"

        pattern = re.compile(directory)
        file_paths = []

        for filename in os.listdir(path):
            if re.search(pattern, filename):
                file_paths.append(path + "/" + filename)

        return file_paths

    sys.exit("Directory not exsist!\nUse \"Apple\" or \"Grape\"")


def analyze_data(image_paths):

    data_infos = {}

    for image_path in image_paths:    
        lst = os.listdir(image_path)
        data_infos[image_path.split("/")[-1]] = len(lst)

    return data_infos


def visualize_data(directory, data_infos):
    
    labels = []
    sizes = []

    for x, y in data_infos.items():
        labels.append(x)
        sizes.append(y)

    # pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.title(directory+" Class Distribution")

    plt.show()

    # bar chart
    plt.bar(labels, sizes)
    plt.title(directory+" Class Distribution")

    plt.show()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help="Directory")

    args = parser.parse_args()
    directory = args.d

    image_paths = get_images(directory)
    data_infos = analyze_data(image_paths)
    visualize_data(directory, data_infos)
