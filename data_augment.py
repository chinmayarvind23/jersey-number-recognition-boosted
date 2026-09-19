import os
import cv2
import numpy as np
import random

_SRCPATH = "./denoised_images"
_DESTPATH = "./augmented_images"

def augment_image_even_odd(image_path, destination_dir):
    """
    Augments an image by setting every other pixel to black, creating even and odd versions,
    and saves one of them randomly with the original filename.

    Args:
        image_path (str): Path to the input image.
        destination_dir (str): Path to the directory to save the augmented image.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return

        rows, cols, channels = img.shape
        augmented = img.copy()

        # Randomly choose even or odd pixel blackout
        if random.choice([True, False]):
            for i in range(rows):
                for j in range(cols):
                    if (i + j) % 2 == 0:
                        augmented[i, j] = [0, 0, 0]
        else:
            for i in range(rows):
                for j in range(cols):
                    if (i + j) % 2 != 0:
                        augmented[i, j] = [0, 0, 0]
            

        base_filename = os.path.basename(image_path)
        destination_path = os.path.join(destination_dir, base_filename)

        cv2.imwrite(destination_path, augmented)
        print(f"Augmented image saved to {destination_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def augment_directory_even_odd(source_dir, destination_dir):
    """
    Augments all images in a source directory and its subdirectories,
    maintaining the subdirectory structure in the destination directory.

    Args:
        source_dir (str): Path to the directory containing input images.
        destination_dir (str): Path to the directory to save augmented images.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for subdir, dirs, files in os.walk(source_dir):
        relative_subdir = os.path.relpath(subdir, source_dir)
        destination_subdir = os.path.join(destination_dir, relative_subdir)

        if not os.path.exists(destination_subdir):
            os.makedirs(destination_subdir)

        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                source_path = os.path.join(subdir, filename)
                augment_image_even_odd(source_path, destination_subdir)


augment_directory_even_odd(_SRCPATH,_DESTPATH)