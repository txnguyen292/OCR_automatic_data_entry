# import the necessary packages
# from pyimagesearch.image_alignment import align_images
import sys
# import Image
import pathlib
from typing import Dict, List, Union, Tuple, NamedTuple, Optional
from config import CONFIG
sys.path.insert(1, str(CONFIG.src))
from alignment_test import alignImages
from collections import namedtuple
import pytesseract
import argparse
import imutils
import cv2
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pimg
from logzero import setup_logger
import logging


loglvl = {"info": logging.INFO, "debug": logging.DEBUG, "warning": logging.WARNING}
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

class OCR:
    """Implement OCR pipeline
    """
    def __init__(self, annotations: str):
        self.annotations: str = annotations
        self.parsingResults: List[str] = []

    # Only use when annotation file gets updated
    def get_unique_files_code(self, file_path: Union[str, pathlib.WindowsPath], columns: Union[List[str], None]=None, header: Tuple[int, None]=None) -> Dict[int, str]:
        """Read annotation file and infer id to file name schema

        Args:
            file_path (Union[str, pathlib.WindowsPath]): [description]
            columns (Union[List[str], None], optional): columns of the data frame. Defaults to None.
            header ([bool], optional): whether to include header or not
        Returns:
            Dict[int, str]: [description]
        """
        if columns is None:
            columns = ["box_name", "x", "y", "w", "h", "image_name", "image_width", "image_height", "stop_words"]
        self.df: pd.DataFrame = pd.read_csv(file_path, header=header)
        self.df.columns: List[str] = columns
        list_of_files = list(self.df.image_name.unique())
        self.id_to_file: Dict[int, str] = {idx: file_name for idx, file_name in enumerate(list_of_files)}

# def test_get_unique_files_code():
#     file_path = CONFIG.data / "annotations" / "final_project.csv"
#     # id_to_file = get_unique_files_code(file_path)
#     assert get_unique_files_code(file_path) == id_to_filename, "Check your id to file_name scheme!"

#--------------------------------------------------------------------------------------------
# This is where classification comes in to output id
# as inputs to get_img_from_id
    def get_img_from_id(self, id: int) -> str:
        """Get img name from given id to file name schema

        Args:
            id (int): [description]

        Returns:
            str: name of the image
        """
        return self.id_to_file[id]

# def test_get_img_from_id():
#     file_path = CONFIG.data / "annotations" / "final_project.csv"
#     id_to_file = get_unique_files_code(file_path)
#     f1040_1 = get_img_from_id(id_to_file, 0)
#     assert f1040_1 == 'f1040--1988-page-001.jpg', "Check your get img function!"

#---------------------------------------------------------------------------------------------

    def cleanup_text(self, text: str) -> str:
        """Strip out non-ASCII text so we can fraw the text on the image"""
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

#==============================================================================================
#----------------------------------------------------------------------------------------------
    def categorize(self, path):
        stringtomatch = pytesseract.image_to_string(cv2.imread(path))
        test = re.findall(re.escape('Filing Status'), stringtomatch)
        if test:
            return 0
        test = re.findall(re.escape('Form 1040 (1988)'), stringtomatch)
        if test:
            return 1
        test = re.findall(re.escape('1545-0139'), stringtomatch)
        if test:
            return 2
        test = re.findall(re.escape('2106 (1988) Page 2'), stringtomatch)
        if test:
            return 3
        test = re.findall(re.escape('1545-0068'), stringtomatch)
        if test:
            return 4
        test = re.findall(re.escape('Profit or Loss From Business'), stringtomatch)
        if test:
            return 5
        test = re.findall(re.escape('1545-0172'), stringtomatch)
        if test:
            return 6
        test = re.findall(re.escape('4562 (1988) Page 2'), stringtomatch)
        if test:
            return 7
        test = re.findall(re.escape('1545-0227'), stringtomatch)
        if test:
            return 8
        test = re.findall(re.escape('Capital Gains and Losses'), stringtomatch)
        if test:
            return 9
        return "Other"
  

# create a named tuple which we can use to create locations of
# the input document which we wish to OCR
    OCRLocation: NamedTuple = namedtuple("OCRLocation", ["id", "bbox", "filter_keywords", "form"])

    def ocr(self, img_name: str, img_reference: str, id: int, debug=False) -> List[str]:
        """Perform ocr to get texts from images

        Args:
            img_name (str): img to perform ocr
            img_reference (str): template image
            id (int): type of form (results from categorize)

        Returns:
            List[str]: OCR'd texts
        """
        # Read in image
        print("Load images...")
        image: np.ndarray = cv2.imread(img_name)
        ih, iw, ic = image.shape
        template: np.ndarray = cv2.imread(img_reference)
        th, tw, tc = template.shape
        # Rescale input image to be equal to template image
        scaled_h, scaled_w = (ih / th), (iw / tw)
        dim = (int(iw / scaled_w), int(ih / scaled_h))
        scaled_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        # aligning input image with template image
        print("Perform alignment...")
        aligned, h = alignImages(scaled_image, template)
        # initialize results as a list
        parsingResults = []
        # get form info
        form = self.id_to_file[id]
        # print(form)
        # Need to replace condition with form later 
        form_info = self.df.loc[self.df.image_name == form, ["box_name", "x", "y", "w", "h"]]
        # print(form_info)
        # regex pattern to remove non-words
        word_pattern = re.compile(r"\w+")
        # instruction words that are not relevant to our results
        # ? Keep this for later use maybe
        # stop_words = "Your first name and initial(if joint return, also give spouse's name and initial), page 6 instructions Present Home Address social security number City Town"
        # stop_words = stop_words.lower().split()
        # print(stop_words)
        stop_words = self.df.loc[self.df.image_name == form, "stop_words"].dropna().to_numpy()
        stop_words = ", ".join(stop_words).lower()
        #?
        # print(stop_words)
        print("Performing OCR on input image...")
        print("Get annotation information...")
        OCR_LOCATIONS = []
        for box_name, x, y, w, h in form_info.values: # Need to isnert stop words here later
            OCR_LOCATIONS.append(OCR.OCRLocation(box_name, (x, y, w, h), stop_words, form))
        if debug:
            fig = plt.figure(figsize=(12, 10))
            rows = cols = np.ceil(np.sqrt(len(OCR_LOCATIONS)))
        for idx, loc in enumerate(OCR_LOCATIONS):
            (x, y, w, h) = loc.bbox
            roi = aligned[y:y+h, x:x+w]
            if debug:
                plt.subplot(rows, cols, idx + 1)
                plt.imshow(roi)
                fig.suptitle(f"Peeking into what tesseract is looking at for form {form}")
            text = pytesseract.image_to_string(roi)
            for line in text.split("\n"):
                # print(line)
                if len(line) == 0 or not word_pattern.match(line):
                    continue
                lower_line = line.lower().split(" ")
                count = sum([lower_line.count(x) for x in stop_words.split(" ")])
                # print(f"{line:8}-{count:8}")
                if count == 0:
                    self.parsingResults.append((loc, line))
    def start(self):
        self.get_unique_files_code(self.annotations)
    def ocr_on_annotations(self):
        pass


if __name__ == "__main__":
    # main()
    pass


