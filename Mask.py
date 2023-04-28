import numpy as np
import cv2
from collections import deque
from datetime import datetime
from ImageProcessing import *
from SaveImg import *

class Mask:
    def __init__(self, timeInterval:int, customMask:bool, customMaskPath:str, downsampleMask:bool, format, updateMask:bool):
        # Public variables .
        self.mCurrentMask = None
        # Private variables .
        self.mOriginalMask = None
        self.mUpdateInterval:int = timeInterval
        self.mUpdateMask:bool = updateMask
        self.mMaskToCreate:bool = False
        self.refDate:str = (datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S')
        self.updateStatus:bool = True
        self.saturatedValue:int = 0
        self.satMap:deque = deque(maxlen=2)

        # Load a mask from file .
        if customMask :
            self.mOriginalMask = cv2.imread(customMaskPath, cv2.IMREAD_GRAYSCALE)
            # Save the mask : to do later 

            # Check if the mask data is loaded successfully
            if self.mOriginalMask is None:
                raise Exception("Fail to load the mask from its path.")
            # Perform pyramid down (downsampling) operation on the mask image
            if downsampleMask :
                self.mOriginalMask = cv2.pyrDown(self.mOriginalMask, dstsize=(self.mOriginalMask.shape[1]//2,self.mOriginalMask.shape[0]//2))
            self.mCurrentMask = self.mOriginalMask.copy()
        else :
            self.mMaskToCreate = True
        
        # Estimate the saturated value .
        if format == 'MONO12':
            self.saturatedValue = 4092
        else :
            self.saturatedValue = 254

    
    def applyMask(self, currFrame):
        # Create a mask .
        if self.mMaskToCreate :
            self.mOriginalMask = np.full((currFrame.shape[0], currFrame.shape[1]), 255, dtype=np.uint8)
            self.mCurrentMask = self.mOriginalMask.copy()
            self.mMaskToCreate = False
        if self.mUpdateMask :
            if self.updateStatus :
                if self.mCurrentMask.shape[0] != currFrame.shape[0] or self.mCurrentMask.shape[1] != currFrame.shape[1] :
                    raise Exception("Mask's size is not correct according to frame's size.")
                # Reference date .
                self.refDate = (datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S')

                # Create the saturatedmap of the current frame .
                saturateMap = buildSaturatedMap(currFrame, self.saturatedValue)

                # Dilatation of the saturated map .
                dilationSize = 10
                # Create a structuring element for dilation .
                element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilationSize + 1, 2*dilationSize + 1), (dilationSize, dilationSize))
                # Perform dilation on the `saturateMap` image using the created structuring element .
                saturateMap = cv2.dilate(saturateMap, element)
                
                
                self.satMap.append(saturateMap)
                if len(self.satMap) == 2 :
                    temp = cv2.bitwise_and(self.satMap[0], self.satMap[1])
                    temp = cv2.bitwise_not(temp)
                    # copy the values of temp to mCurrentMask image, but only where mOriginalMask is non-zero .
                    self.mCurrentMask[self.mOriginalMask != 0] = temp[ self.mOriginalMask != 0]
                    # self.mCurrentMask = temp.copy()
                
                # copy the values from currFrame to temp image, but only where mCurrentMask is non-zero .
                temp = np.zeros(currFrame.shape, dtype=np.uint8)
                temp[self.mCurrentMask != 0] = currFrame[self.mCurrentMask != 0]
                SaveImg.saveJPEG(temp, "saturation")
                # temp = currFrame.copy()
                currFrame = temp.copy()

                self.updateStatus = False
                return (True, currFrame) # Mask not applied, only computed .
            
            nowDate = (datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S')
            t1 = datetime.strptime(self.refDate, "%Y-%m-%d %H:%M:%S")
            t2 = datetime.strptime(nowDate, "%Y-%m-%d %H:%M:%S")
            td = t2 - t1
            diffTime = td.total_seconds() 
            if(diffTime >= self.mUpdateInterval) :
                self.updateStatus = True
            print(f"Next mask : {self.mUpdateInterval - int(diffTime)}")

        if self.mCurrentMask.all() == None or (self.mCurrentMask.shape[0] != currFrame.shape[0] and self.mCurrentMask.shape[1] != currFrame.shape[1]):
            self.mMaskToCreate = True
            return (True, currFrame)
        # Copy the pixels drom currFrame to temp only where mCurrentMask is not zero .
        temp = cv2.copyTo(currFrame, None, self.mCurrentMask)
        currFrame = temp.copy()

        return (False, currFrame) # Mask applied

    def resetMask(self):
        self.mCurrentMask = self.mOriginalMask
        self.updateStatus = True
        self.satMap.clear()