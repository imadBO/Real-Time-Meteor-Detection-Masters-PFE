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
    lut_matrix = np.array((np.arange(256) / 255.0) ** (1 / gamma) * 255.0, dtype=np.uint8)
    result = cv2.LUT(img, lut_matrix)
    return result

def correctGammaOnMono12(img, gamma):
    """
    Gamma correction is a non-linear operation that adjusts the intensity values of an image to correct
    for the non-linear response of displays or sensors. It is commonly used to correct the brightness or 
    contrast of an image. The gamma correction formula is typically applied to each pixel value in the image
    and is given by:
    Corrected_Value = (Original_Value / 4095.0) ^ (1 / gamma) * 4095.0
    """
    gamma_inverse = 1.0 / gamma
    result = ((img / 4095.0) ** gamma_inverse * 4095.0).astype(np.uint16)
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

    for _ in range(1, n * n):

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

    mean, stddev = cv2.meanStdDev(img, mask=mask)
    threshold = 0

    if threshType == 'MEAN':
        threshold = np.mean(mean) * factor + 10
    elif threshType == 'STDEV':
        threshold = np.mean(stddev) * factor + 10

    thresholdedMap = np.mean(img, axis=2) > threshold
    thresholdedMap = thresholdedMap.astype(np.uint8) * 255

    return thresholdedMap

def buildSaturatedMap(img, maxval):
    """
    This function returns a binary map indicating saturated pixels in the input image.
    If a pixel value is greater than or equal to maxval, the corresponding pixel value 
    in the saturatedMap is set to 255, indicating a saturated pixel.
    """
    # create saturated_map with the same shape and dtype as img .
    saturated_map = np.zeros_like(img, dtype=np.uint8)
    # We calculate the mean of each pixel using np.mean(img, axis=2). By specifying axis=2, we compute the mean across the color channels, resulting in a 2D array with shape (img.shape[0], img.shape[1]).
    # We create a boolean mask saturated_pixels indicating which pixels are saturated by comparing the mean values to maxval.
    saturated_pixels = np.mean(img, axis=2) >= maxval
    # We assign the value 255 to the saturated pixels in saturated_map using boolean indexing saturated_map[saturated_pixels] = 255
    saturated_map[saturated_pixels] = 255
    return saturated_map