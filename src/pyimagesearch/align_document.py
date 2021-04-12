"""Load image and template and perform image alignment"""
# import the necessary packages

from image_alignment import align_images
import numpy as np
import argparse
import imutils
import cv2
from pathlib import Path

# construct the argument parser and parse the arguments
def get_args():
    parser = argparse.ArgumentParser(
        description="Perform image alignment with input image and template",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--image", required=True,
                        help="path to input image for alignment")
    parser.add_argument("-t", "--template", required=True,
                        help="path to input template image")
    parser.add_argument("-o", "--output", required=False,
                        help="Path to save aligned image")
    return vars(parser.parse_args())
#----------------------------------------------------------------------------

if __name__ == "__main__":
    args = get_args()

    # load the input image
    print("[INFO] loading images...")
    image = cv2.imread(args["image"])
    template = cv2.imread(args["template"])

    # align the images
    print("[INFO] aligning images...")
    aligned = align_images(image, template)

    # img_name = Path(args["image"]).name
    # print(str(img_name))
    # path_to_save = Path(args["output"]) / ("aligned"+img_name)
    # cv2.imwrite(str(path_to_save), aligned)

    aligned = imutils.resize(aligned, width=700)
    template = imutils.resize(template, width=700)

    stacked = np.hstack([aligned, template])
    cv2.imshow("Image Alignment Stacked", stacked)
    cv2.waitKey(0)
