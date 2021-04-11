""" Implement image alignment using feature-based method with opencv"""
from config import CONFIG

import numpy as np
import imutils
import cv2
import logging
from config import CONFIG
from logzero import setup_logger


logger = setup_logger(__file__, level=logging.INFO, logfile=str(CONFIG.report / "image_aligment.log"))

def align_images(image: np.ndarray, template: np.ndarray, 
                maxFeatures: int=500, keepPercent: float=0.2,
                debug: bool=False):

    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_h, input_w = imageGray.shape
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    temp_h, temp_w = templateGray.shape
    # rescaling images for optimal alignment
    scaled_h, scaled_w = input_h / temp_h, input_w / temp_w
    dim = (int(input_w / scaled_w), int(input_h / scaled_h))
    logger.info(f"Original input image shape {input_h, input_w}")
    logger.info(f"Template image shape {temp_h, temp_w}")
    logger.info(f"Rescale input image shape to {dim}")
    imageGray = cv2.resize(imageGray, dim, interpolation=cv2.INTER_AREA)
    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descB) = orb.detectAndCompute(templateGray, None)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descA, descB, None)

    # sort the matches by their distance (the smaller the 
    # distance, the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)

    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
        matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)
    
    # allocate memory for the keypoints (x, y) - coordinates from
    # the top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    # return the aligned image
    return aligned

if __name__ == "__main__":
    pass