import os
import shutil
import cv2
import random
import gc

OG_FOLDER = "..\OG_Dataset"

IMAGE_FOLDERS = ["DIV2K_train_HR", "DIV2K_valid_HR", "Flickr2K"]

local_folder = "images"

IMG_SIZE = 256
start = 0


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


create_folder(local_folder)


for folder in IMAGE_FOLDERS:
    folder_path = os.path.join(OG_FOLDER, folder)
    folder_ = os.listdir(folder_path)
    length = len(folder_)
    # print(folder_path, length)
    # continue
    for idx, image_name in enumerate(sorted(folder_)):
        image_path = os.path.join(folder_path, image_name)
        raw_image = cv2.imread(image_path)

        new_image_name = f"{start:05}.png"
        bicubic_image = cv2.resize(raw_image, (IMG_SIZE, IMG_SIZE), cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(local_folder, new_image_name), bicubic_image)

        start += 1
        del raw_image, bicubic_image
        gc.collect()
        print(f"{folder} -> Images Completed: {idx+1}/{length}", end="\r")
    print()
