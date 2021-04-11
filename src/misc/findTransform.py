import cv2
from cv2 import countNonZero, cvtColor
import numpy as np
from config import CONFIG

def align_images_with_opencv(path_to_desired_image: str, path_to_image_to_warp: str,
                             output_path: str):
    # See https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/

    # image_sr_path = "C:/Users/Alex/Repositories/CVC-MUSCIMA/CVCMUSCIMA_SR/CvcMuscima-Distortions/ideal/w-50/image/p020.png"
    # image_wi_path = "C:/Users/Alex/Repositories/CVC-MUSCIMA/CVCMUSCIMA_WI/CVCMUSCIMA_WI/PNG_GT_Gray/w-50/p020.png"

    # Read the images to be aligned
    im1 = cv2.imread(path_to_desired_image)
    im2 = cv2.imread(path_to_image_to_warp)

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1_gray = cv2.bitwise_not(im1_gray)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_AFFINE

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 100

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-7

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
    except TypeError:
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    # Warp BW-image
    bw_image_path = str.replace(path_to_image_to_warp, "PNG_GT_Gray", "PNG_GT_BW")
    bw_image = cv2.imread(bw_image_path)
    bw_image_gray = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
    bw_image_aligned = cv2.warpAffine(bw_image_gray, warp_matrix, (sz[1], sz[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(bw_image_path, bw_image_aligned)

    # Warp NoStaff-image
    bw_no_staff_image_path = str.replace(path_to_image_to_warp, "PNG_GT_Gray", "PNG_GT_NoStaff")
    bw_no_staff_image = cv2.imread(bw_no_staff_image_path)
    bw_no_staff_image_gray = cv2.cvtColor(bw_no_staff_image, cv2.COLOR_BGR2GRAY)
    bw_no_staff_image_aligned = cv2.warpAffine(bw_no_staff_image_gray, warp_matrix, (sz[1], sz[0]),
                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(bw_no_staff_image_path, bw_no_staff_image_aligned)

    # Warp Gray-Image last, in case user interrupts, to make sure the process continues appropriately and doesn't
    # miss the bw and bw_no_staff images, because the process compares only on grayscale images
    im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1], sz[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    cv2.imwrite(output_path, im2_aligned)


if __name__ == "__main__":
    align_images_with_opencv(str(CONFIG.data / "imgs" / "f1040--1988-1.png"), str(CONFIG.data / "test_img.png"), str(CONFIG.data / "data"))
