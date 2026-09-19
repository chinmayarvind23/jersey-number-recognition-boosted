import os
import cv2
import numpy as np

_SRCPATH = "./data/SoccerNet/test/images"
_DESTPATH = "denoised_images"

def denoise_image(source_path, destination_path, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    """
    Denoises an image using the Non-local Means Denoising algorithm.

    Args:
        source_path (str): Path to the noisy image.
        destination_path (str): Path to save the denoised image.
        h (int): Parameter regulating filter strength. Higher h value removes noise better, but removes details of an image also.
        hColor (int): Parameter regulating filter strength for color components. A value of 10 is good enough in most cases.
        templateWindowSize (int): Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels.
        searchWindowSize (int): Size in pixels of the search window that is used to compute weighted average. Should be odd. Recommended value 21 pixels.
    """
    try:
        img = cv2.imread(source_path)
        if img is None:
            print(f"Error: Could not read image from {source_path}")
            return

        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)
        cv2.imwrite(destination_path, denoised_img)
        return destination_path

    except Exception as e:
        print(f"An error occurred: {e}")

def denoise_directory(source_dir, destination_dir):
    """
    Denoises all images in a source directory and its subdirectories,
    maintaining the subdirectory structure in the destination directory.

    Args:
        source_dir (str): Path to the directory containing noisy images.
        destination_dir (str): Path to the directory to save denoised images.

    Returns:
        destination_dir
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for subdir, dirs, files in os.walk(source_dir):
        relative_subdir = os.path.relpath(subdir, source_dir)  # Get relative path
        destination_subdir = os.path.join(destination_dir, relative_subdir)

        if not os.path.exists(destination_subdir):
            os.makedirs(destination_subdir)

        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                source_path = os.path.join(subdir, filename)
                destination_path = os.path.join(destination_subdir, filename)
                denoise_image(source_path, destination_path)
                

    return destination_dir

#CHANGE THE SOURCE DIRECTORY ACCORDINGLY
denoise_directory(_SRCPATH, _DESTPATH) 