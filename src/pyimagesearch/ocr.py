# import the necessary packages
# from pyimagesearch.image_alignment import align_images
import sys
from config import CONFIG
sys.path.insert(1, str(CONFIG.src))
# print(sys)
from alignment_test import alignImages
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2
from logzero import setup_logger
import logging


loglvl = {"info": logging.INFO, "debug": logging.DEBUG, "warning": logging.WARNING}

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
    parser.add_argument("-llvl", "--loglvl", required=False,
                        help="set the level of logging", default="info")
    
    return vars(parser.parse_args())

# create a named tuple which we can use to create locations of
# the input focument which we wish to OCR

OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords"])
OCR_LOCATIONS = [
    OCRLocation("step1_name", (250,118,981,66), ["middle", "initial", "first", "name"]),
    OCRLocation("step1_address", (316,181,910,74), ["Present", "home", "address"]),
    OCRLocation("step1_city_state_zip", (316,249,913,71), ["city", "zip", "town", "state"]),
    OCRLocation("step4_SSN", (1221,121,321,66), ["social","security","number"]),
    OCRLocation("step5_total_income", (1275,1609,278,38), [""])
]




if __name__ == "__main__":
    args = get_args()
    logger = setup_logger(__file__, level=loglvl[args["loglvl"]], logfile=str(CONFIG.report / "ocr.log"))
    logger.info("Loading images...")
    image = cv2.imread(args["image"])
    ih, iw, ic = image.shape
    template = cv2.imread(args["template"])
    th, tw, tc = template.shape
    scaled_h, scaled_w = (ih / th), (iw / tw)
    dim = (int(iw / scaled_w), int(ih / scaled_h))
    scaled_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # align the images
    logger.info("aligning images...")
    aligned, h = alignImages(image, template)

    logger.info("OCR'ing document ...")
    parsingResults = []

    # loop over the locations of the document we are going to OCR
    for loc in OCR_LOCATIONS:
        # extract the OCR ROI from the aligned image
        (x, y, w, h) = loc.bbox
        roi = aligned[y:y+h, x:x+w]

        # OCR the ROI using Tesseract
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(rgb)
        # break the text into lines and loop over 
        for line in text.split("\n"):
            # if the line is empty, ignore it
            if len(line) == 0:
                continue
            # convert the line to lowercase and then check to see
            # if the line contains any of the filter keywords (these
            # keywords are part of the *form itself* and should be ignored) 

            lower = line.lower()
            count = sum([lower.count(x) for x in loc.filter_keywords])

            # if the count is zero then we know we are *not* examining
            # a text field that is part of the document itself (ex., info,
            # on the field, an example, help text, etc.)
            if count == 0:
                # update our parsing results dictionary with the OCR'd
                # text if the line is *not* empty
                parsingResults.append((loc, line))
        # initialize a dictioanry to store our final OCR results
        results = {}

        # loop over the results of parsing the document
        for (loc, line) in parsingResults:
            # grab any existing OCR result for the current ID of the document
            r = results.get(loc.id, None)
            # if the result is None, initialize it using the text and location
            # namedtuple (converting it to a dictionary as namedtuples are not hashable)
            if r is None:
                results[loc.id] = (line, loc._asdict())
            
            # otherwise, there exists an OCR result for the current area of the document
            # so we should append our exising line
            else:
                # unpack the existing OCR result and append the line to 
                # the existing text
                (existingText, loc) = r
                text = f"{existingText}\n{line}"

                # update our results dictionary
                results[loc["id"]] = (text, loc)
        # loop over the results
        for (locID, result) in results.items():
            # unpack the result tuple
            (text, loc) = result
            # display the OCR result to our terminal
            print(loc["id"])
            print("=" * len(loc["id"]))
            print(f"{text}\n\n")

