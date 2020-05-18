'''organize the data into ImageFolder format '''

import pandas as pd
import os
from argparse import ArgumentParser
from p_tqdm import p_map
import subprocess


def get_files(root_dir, ext_length):
    '''get the list of files'''
    # only return if no subdirectory
    if len(os.listdir(root_dir)) == 0:
        # remove empty directory
        subprocess.call(["rm", "-rf", os.path.abspath(root_dir)])
        return {}
    if not any([os.path.isfile(os.path.join(root_dir, file)) for file in os.listdir(root_dir)]):
        subprocess.call(["rm", "-rf", os.path.abspath(root_dir)])
        return {}
    else:
        files = {}
        for file in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file)
            if not os.path.isfile(file_path):
                files.update(get_files(file_path, ext_length))
        return files


def move_image(img_obj, output_dir="/data/google-landmark/org/train"):
    '''organize images in one single subdirectory'''

    file_path = img_obj['path']
    label = img_obj['label']

    # create the folder if not there
    label_dir = os.path.abspath(os.path.join(output_dir, str(label)))

    # move the image
    subprocess.call(["mv", file_path, label_dir])


if __name__ == "__main__":

    argparse = ArgumentParser("Organize the data into Image Folder format")

    argparse.add_argument("-c", "--csv_file", help="csv file containing instance to label mapping",
                          dest="csv_file", default="/data/google-landmark/csv/train.csv")
    argparse.add_argument("-i", "--root_dir", help="root image folder", dest="root_dir",
                          default="/data/google-landmark/train")
    argparse.add_argument("-o", "--output_idr", help="output folder", dest="output_dir",
                          default="/data/google-landmark/org/train")
    argparse.add_argument("-e", "--extension", help="file extension. default: jpg", dest="extension",
                          default="jpg")

    arg = argparse.parse_args()
    csv_file = arg.csv_file
    output_dir = arg.output_dir
    root_dir = arg.root_dir

    img_ids = pd.read_csv(csv_file).id.to_list()
    label_ids = pd.read_csv(csv_file).landmark_id.to_list()

    for label in label_ids:
        label_dir = os.path.abspath(os.path.join(output_dir, str(label)))

        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
            print("directory created %s" % label_dir)

    # get all subdirectories
    img_files = get_files(root_dir, ext_length=len(arg.extension))
    img_objs = []

    for i in range(len(img_ids)):
        if img_ids[i] not in img_files:
            print(img_ids[i])
        else:
            img_objs.append({'path': img_files[img_ids[i]], 'label': label_ids[i]})

    p_map(move_image, img_objs)
