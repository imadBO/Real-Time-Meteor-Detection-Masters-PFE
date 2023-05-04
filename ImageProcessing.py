import cv2
import numpy as np
from Types import Point
def correctGammaOnMono8(img, gamma):
    """
    Gamma correction is a non-linear operation that adjusts the intensity values of an image to correct
    for the non-linear response of displays or sensors. It is commonly used to correct the brightness or 
    contrast of an image. The gamma correction formula is typically applied to each pixel value in the image
    and is given by:
    Corrected_Value = (Original_Value / 255.0) ^ (1 / gamma) * 255.0
    """
    gamma_inverse = 1.0 / gamma

    lut_matrix = np.zeros((1, 256), dtype=np.uint8)
    for i in range(256):
        lut_matrix[0, i] = int((i / 255.0) ** gamma_inverse * 255.0)

    result = cv2.LUT(img, lut_matrix)

    return result

def correctGammaOnMono12(img, gamma):
    """
    Gamma correction is a non-linear operation that adjusts the intensity values of an image to correct
    for the non-linear response of displays or sensors. It is commonly used to correct the brightness or 
    contrast of an image. The gamma correction formula is typically applied to each pixel value in the image
    and is given by:
    Corrected_Value = (Original_Value / 255.0) ^ (1 / gamma) * 255.0
    """
    gamma_inverse = 1.0 / gamma

    result = np.zeros_like(img, dtype=np.uint16)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = int((img[i, j] / 4095.0) ** gamma_inverse * 4095.0)

    return result

def subdivideFrame(n, imgH, imgW):
    """
    This function subdivides a frame into n x n equal-sized subdivisions and store the positions of 
    these subdivisions in the sub vector of Point objects .

    Example : frame with n = 4 -> 16 subdivisions returned

    |07|08|09|10|
    |06|01|02|11|
    |05|04|03|12|
    |16|15|14|13|
    """
    # The array which will store the subdivided frame positions .
    sub = []
    # The size of each subdivision, where n is the number of subdivisions needed
    # in each dimension, imgH and imgW are the frame height and width .
    subW = imgW // n
    subH = imgH // n

    # The starting position of the subdivisions, the top-left corner of the subdivision . 
    first = Point(x=(n // 2 - 1) * subW, y=(n // 2) * subH)
    # last = Point(x=imgW - subW, y=imgH - subH)

    sub.append(first)

    x, y = first.x, first.y
    nbDirection = 0 # The number of subdivisions made in the current direction .
    nbDirectionLimit = 1 # The limit of subdivisions in that direction .
    direction = 1  # 1 up
                   # 2 right
                   # 3 down
                   # 4 left

    for i in range(1, n * n):

        if direction == 1:

            y = y - subH
            sub.append(Point(y,x))
            nbDirection += 1
            if nbDirection == nbDirectionLimit:
                nbDirection = 0
                direction += 1

        elif direction == 2:

            x = x + subW
            sub.append(Point(y,x))
            nbDirection += 1
            if nbDirection == nbDirectionLimit:
                nbDirection = 0
                nbDirectionLimit += 1
                direction += 1

        elif direction == 3:

            y = y + subH
            sub.append(Point(y,x))
            nbDirection += 1
            if nbDirection == nbDirectionLimit:
                nbDirection = 0
                direction += 1

        elif direction == 4:

            x = x - subW
            sub.append(Point(y,x))
            nbDirection += 1
            if nbDirection == nbDirectionLimit:
                nbDirection = 0
                nbDirectionLimit += 1
                direction = 1

    return np.array(sub)

def thresholding(img, mask, factor, threshType):
    """
    This function performs thresholding on a frame (img) based on a given mask (mask) and thresholding factor (factor) 
    using a specified thresholding method (threshType: mean or stdev)
    """

    thresholdedMap = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    # Compute the mean and stdev of the frame, and use the optional mask to only consider 
    # the pixels that are non-zero in the mask .
    mean, stddev = cv2.meanStdDev(img, mask=mask)
    # print(mean, stddev)
    # Initializes the threshold value to 0.
    threshold = 0

    # Calculate the value of the threshold based on mean or stdev and a factor .
    if threshType == 'MEAN':
        threshold = np.mean(mean) * factor
    elif threshType == 'STDEV':
        threshold = np.mean(stddev) * factor

    if img.dtype == np.uint16:
        if threshold == 0:
            threshold = 65535

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.mean(img[i, j]) > threshold:
                    thresholdedMap[i, j] = 255

    elif img.dtype == np.uint8:
        if threshold == 0:
            threshold = 255

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.mean(img[i, j]) > threshold:
                    thresholdedMap[i, j] = 255

    return thresholdedMap

def buildSaturatedMap(img, maxval):
    """
    This function returns a binary map indicating saturated pixels in the input image.
    If a pixel value is greater than or equal to maxval, the corresponding pixel value 
    in the saturatedMap is set to 255, indicating a saturated pixel.
    """
    saturated_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    

    if img.dtype == np.uint16:

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.mean(img[i,j]) >= maxval:
                    saturated_map[i,j] = 255
    elif img.dtype == np.uint8:

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.mean(img[i,j]) >= maxval:
                    saturated_map[i,j] = 255
    return saturated_map