# import the necessary packages
# from pyimagesearch.image_alignment import align_images
import sys
import pathlib
from typing import Dict, List, Union, Tuple, NamedTuple
from config import CONFIG
sys.path.insert(1, str(CONFIG.src))
# print(sys)
from alignment_test import alignImages
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2
import pandas as pd
from logzero import setup_logger
import logging


loglvl = {"info": logging.INFO, "debug": logging.DEBUG, "warning": logging.WARNING}
id_to_filename = {0: 'f1040--1988-page-001.jpg',
        1: 'f1040--1988-page-002.jpg',
        2: 'f2106--1988-page-001.jpg',
        3: 'f2106--1988-page-002.jpg',
        4: 'f2441--1988-page-001.jpg',
        5: 'f1040sc--1988-page-001.jpg',
        6: 'f4562--1988-page-001.jpg',
        7: 'f4562--1988-page-002.jpg',
        8: 'f6251--1988-page-001.jpg',
        9: 'f1040sd--1988-page-001.jpg'
        }
#-------------------------------------------------------------------------------------
def get_args():
    """Get command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Perform OCR on image of forms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-annot", "--annotation", required=True,
                        help="path to annotation file")

    parser.add_argument("-o", "--output", required=True,
                        help="path to output OCR'd texts")
    parser.add_argument("-llvl", "--loglvl", required=False,
                        help="set the level of logging", default="info")
    
    return vars(parser.parse_args())

#=====================================================================================
#========== HELPER FUNCTIONS =========================================================
#=====================================================================================

# Only use when annotation file gets updated
def get_unique_files_code(file_path: Union[str, pathlib.WindowsPath]) -> Dict[int, str]:
    """Read annotation file and infer id to file name schema"""
    columns = ["box_name", "x", "y", "w", "h", "image_name", "image_width", "image_height", "stop_words"]
    df = pd.read_csv(file_path, header=None)
    df.columns = columns
    list_of_files = list(df.image_name.unique())
    id_to_file = {idx: file_name for idx, file_name in enumerate(list_of_files)}
    return id_to_file

def test_get_unique_files_code():
    file_path = CONFIG.data / "annotations" / "final_project.csv"
    # id_to_file = get_unique_files_code(file_path)
    res = {0: 'f1040--1988-page-001.jpg',
            1: 'f1040--1988-page-002.jpg',
            2: 'f2106--1988-page-001.jpg',
            3: 'f2106--1988-page-002.jpg',
            4: 'f2441--1988-page-001.jpg',
            5: 'f1040sc--1988-page-001.jpg',
            6: 'f4562--1988-page-001.jpg',
            7: 'f4562--1988-page-002.jpg',
            8: 'f6251--1988-page-001.jpg',
            9: 'f1040sd--1988-page-001.jpg'
            }
    assert get_unique_files_code(file_path) == res, "Check your id to file_name scheme!"
#--------------------------------------------------------------------------------------------
# This is where classification comes in to output id
# as inputs to get_img_from_id
def get_img_from_id(id_to_file: Dict[int, str], id: int) -> str:
    """Get img path from given id to file name schema"""
    return id_to_file[id]

def test_get_img_from_id():
    file_path = CONFIG.data / "annotations" / "final_project.csv"
    id_to_file = get_unique_files_code(file_path)
    f1040_1 = get_img_from_id(id_to_file, 0)
    assert f1040_1 == 'f1040--1988-page-001.jpg', "Check your get img function!"

#---------------------------------------------------------------------------------------------

def cleanup_text(text: str) -> str:
    """Strip out non-ASCII text so we can fraw the text on the image"""
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

#==============================================================================================
#----------------------------------------------------------------------------------------------

# create a named tuple which we can use to create locations of
# the input document which we wish to OCR

OCRLocation = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords"])
# OCR_LOCATIONS = [
#     OCRLocation("step1_name", (250,118,981,66), ["middle", "initial", "first", "name"]),
#     OCRLocation("step1_address", (316,181,910,74), ["Present", "home", "address"]),
#     OCRLocation("step1_city_state_zip", (316,249,913,71), ["city", "zip", "town", "state"]),
#     OCRLocation("step4_SSN", (1221,121,321,66), ["social","security","number"]),
#     OCRLocation("step5_total_income", (1275,1609,278,38), [""])
# ]

def ocr(img_name: str, img_reference: str, id: int, df: pd.DataFrame) -> List[str]:
    """Perform ocr to get texts from images

    Args:
        img_name (str): img to perform ocr
        img_reference (str): template image
        id (int): type of form

    Returns:
        List[str]: OCR'd texts
    """
    # Read in image
    image = cv2.imread(img_name)
    ih, iw, ic = image.shape
    template = cv2.imread(img_reference)
    th, tw, tc = template.shape
    # Rescale input image to be equal to template image
    scaled_h, scaled_w = (ih / th), (iw / tw)
    dim = (int(iw / scaled_w), int(ih / scaled_h))
    scaled_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # aligning input image with template image
    aligned, h = alignImages(image, template)
    # initialize results as a list
    parsingResults = []
    # get form info
    form = id_to_filename[id]
    form_info = df.loc[df.image_name == form, ["box_name", "x", "y", "w", "h", "stop_words"]]
    OCRLocations = []
    for box_name, x, y, w, h, stop_word in form_info.values:
        OCRLocations.append(OCRLocation(box_name, (x, y, w, h), stop_word))
    return None



#-----------------------------------------------------------------------------------------------------------
#===========================================================================================================
def main():
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


if __name__ == "__main__":
    # main()
    file_path = CONFIG.data / "annotations" / "final_project.csv"
    id_to_file = get_unique_files_code(file_path)
    ####
    df = pd.read_csv(file_path, header=None)
    columns = ["box_name", "x", "y", "w", "h", "image_name", "image_width", "image_height", "stop_words"]
    df.columns = columns
    # print(df)
    form_info = df.loc[df.image_name == id_to_file[0], ["box_name", "x", "y", "w", "h", "stop_words"]]
    OCR_LOCATIONS = []
    for box_name, x, y, w, h, stop_word in form_info.values:
        OCR_LOCATIONS.append(OCRLocation(box_name, (x, y, w, h), stop_word))
    for loc in OCR_LOCATIONS:
        # extract the OCR ROI from the aligned image
        (x, y, w, h) = loc.bbox
        roi = aligned[y:y+h, x:x+w]

        # OCR the ROI using Tesseract
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(rgb)
        print(text)


