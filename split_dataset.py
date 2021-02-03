import gc
import os
import shutil
from sklearn.model_selection import train_test_split

ROOT_DIR = r"images_og_2"
TRAIN = r"images_2/train"
VALID = r"images_2/valid"


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def copy_files(source, destination, files):
    create_folder(destination)
    len_ = len(files)
    for idx, image_name in enumerate(files):
        image_path = os.path.join(source, image_name)
        dest_path = os.path.join(destination, image_name)
        try:
            shutil.copy(image_path, dest_path)
        except:
            print(image_path)
            print(dest_path)

        if not (idx + 1) % 50 or idx + 1 == len_:
            print(f"Copied: {idx+1}/{len_} images", end="\r")

    print()
    gc.collect()
    return


if __name__ == "__main__":

    image_names = os.listdir(ROOT_DIR)
    length = len(image_names)

    images_train, images_valid = train_test_split(
        image_names, shuffle=True, random_state=41, test_size=64
    )

    print("Train images:", len(images_train), "Validation images:", len(images_valid))

    copy_files(ROOT_DIR, TRAIN, images_train)
    copy_files(ROOT_DIR, VALID, images_valid)
