# import the necessary packages
from pyimagesearch.image_alignment import align_images
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2

def cleanup_text(text: str) -> str:
    """Strip out non-ASCII text so we can fraw the text on the image"""
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def get_args():
    parser = argparse.ArgumentParser(
        description="Perform OCR on image of forms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--image", required=True,
                        help="path to input image that we'll align to template")

    parser.add_argument("-t", "--template", required=True,
                        help="path to input template image")
    return vars(parser.parse_args())

# create a named tuple which we can use to create locations of
# the input focument which we wish to OCR

OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords"])
OCT_LOCATIONS = []