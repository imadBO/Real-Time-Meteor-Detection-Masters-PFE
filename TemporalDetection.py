import logging
from logging import LogRecord
from logging.handlers import RotatingFileHandler
import numpy as np
import GlobalEvent
from GlobalEvent import *
from Frame import Frame
from Types import Point
# import Mask
from Mask import *
import SParam
from SParam import *
import ECamPixFmt
from ECamPixFmt import *
import pathlib
import SaveImg
from SaveImg import *
from Types import Point
import TimeDate
from TimeDate import *
from ImageProcessing import *
import time


class TemporalDetection:

    logger = logging.getLogger("DetectionTemporal")
    logger.setLevel(logging.DEBUG)

    class Init:
        def __init__(self):
            TemporalDetection.logger.addFilter(logging.Filter(name="ClassName", record=None))
            
    def __init__(self, dtp:detectionParam, fmt:CamPixFmt):
        self.mListGlobalEvents = np.array([], dtype= GlobalEvent) # List of global events (Events spread on several frames) .    
        self.mSubdivisionPos = np.array([], dtype=Point) # Position (origin in top left) of 64 subdivisions .
        self.mListColors = np.array([]) # One color per local event .
        self.mLocalMask = None # Mask used to remove isolated white pixels .
        self.mSubdivisionStatus = False # If subdivisions positions have been computed .
        self.mPrevThresholdedMap = None 
        self.mGeToSave:GlobalEvent = None # Global event to save .
        self.mRoiSize = [10, 10]
        self.mImgNum = 0 # Current frame number .
        self.mPrevFrame = np.array([None]) #  Previous frame .
        self.mStaticMask = None
        self.mDebugCurrentPath = ""
        self.mDataSetCounter = 0
        self.mDebugUpdateMask = False
        self.mMaskManager:Mask = Mask.Mask(timeInterval=dtp.DET_UPDATE_MASK_FREQUENCY, customMask=dtp.ACQ_MASK_ENABLED, customMaskPath=dtp.ACQ_MASK_PATH, downsampleMask=dtp.DET_DOWNSAMPLE_ENABLED, format=fmt, updateMask=dtp.DET_UPDATE_MASK)
        self.debugFiles = np.array([], dtype=str)
        self.mdtp = dtp
        self.mVideoDebugAutoMask = None

        self.mListColors = np.append(self.mListColors, np.array([139, 0, 0], dtype=np.uint8))       # DarkRed
        self.mListColors = np.append(self.mListColors, np.array([255, 0, 0], dtype=np.uint8))       # Red
        self.mListColors = np.append(self.mListColors, np.array([100, 100, 0], dtype=np.uint8))     # IndianRed
        self.mListColors = np.append(self.mListColors, np.array([205, 92, 92], dtype=np.uint8))     # Salmon
        self.mListColors = np.append(self.mListColors, np.array([255, 140, 0], dtype=np.uint8))     # DarkOrange
        self.mListColors = np.append(self.mListColors, np.array([210, 105, 30], dtype=np.uint8))    # Chocolate
        self.mListColors = np.append(self.mListColors, np.array([255, 255, 0], dtype=np.uint8))     # Yellow
        self.mListColors = np.append(self.mListColors, np.array([240, 230, 140], dtype=np.uint8))   # Khaki
        self.mListColors = np.append(self.mListColors, np.array([255, 255, 224], dtype=np.uint8))   # LightYellow
        self.mListColors = np.append(self.mListColors, np.array([148, 0, 211], dtype=np.uint8))     # DarkViolet
        self.mListColors = np.append(self.mListColors, np.array([255, 20, 147], dtype=np.uint8))    # DeepPink
        self.mListColors = np.append(self.mListColors, np.array([255, 0, 255], dtype=np.uint8))     # Magenta
        self.mListColors = np.append(self.mListColors, np.array([0, 100, 0], dtype=np.uint8))       # DarkGreen
        self.mListColors = np.append(self.mListColors, np.array([128, 128, 0], dtype=np.uint8))     # Olive
        self.mListColors = np.append(self.mListColors, np.array([0, 255, 0], dtype=np.uint8))       # Lime
        self.mListColors = np.append(self.mListColors, np.array([127, 255, 212], dtype=np.uint8))   # Aquamarine
        self.mListColors = np.append(self.mListColors, np.array([64, 224, 208], dtype=np.uint8))    # Turquoise
        self.mListColors = np.append(self.mListColors, np.array([0, 0, 205], dtype=np.uint8))       # Blue
        self.mListColors = np.append(self.mListColors, np.array([0, 191, 255], dtype=np.uint8))     # DeepSkyBlue
        self.mListColors = np.append(self.mListColors, np.array([0, 255, 255], dtype=np.uint8))     # Cyan

        # Create local mask to eliminate single white pixels .
        maskTemp = np.full((3, 3), 255, dtype=np.uint8) # create a 3x3 array filled with 255 .
        maskTemp[1, 1] = 0 # set the value at row 1, column 1 to 0 .
        self.mLocalMask = maskTemp.copy() # create a copy of the maskTemp array and assign it to mLocalMask .

        self.mdtp.DET_DEBUG_PATH = self.mdtp.DET_DEBUG_PATH + "/"
        self.mDebugCurrentPath = self.mdtp.DET_DEBUG_PATH

        if dtp.DET_DEBUG:
            self.createDebugDirectories(True)


    # def initMethod(self, cfgPath):
    #     pass

    def runDetection(self, cframe:Frame):
        if self.mSubdivisionStatus == False :
            h = cframe.mImg.shape[0]
            w = cframe.mImg.shape[1]

            if self.mdtp.DET_DOWNSAMPLE_ENABLED :
                h /= 2
                w /= 2

            self.mSubdivisionPos = subdivideFrame(n=8,imgH=h,imgW=w)
            self.mSubdivisionStatus = True
            if self.mdtp.DET_DEBUG :
                s = np.zeros((h,w),dtype=np.uint8)
                for i in range(8):
                    cv2.line(s, (0, int(i * (h/8))), (w - 1, int(i * (h/8))), (255), 1)
                    cv2.line(s, (int(i * (w/8)), 0), (int(i * (w/8)), h-1), (255), 1)
                
                SaveImg.saveBMP(s, self.mDebugCurrentPath + "subdivisions_map")
        else :
            tDownsample  = 0
            tAbsDiff     = 0
            tPosDiff     = 0
            tNegDiff     = 0
            tDilate      = 0
            tThreshold   = 0
            tStep1       = 0
            tStep2       = 0
            tStep3       = 0
            tStep4       = 0
            tTotal       = time.perf_counter()

            ###########################################################################
            #                STEP 1 : FILETRING / THRESHOLDING                        #
            ###########################################################################
            tStep1 = time.perf_counter()
            currImg = None
            # Downsample current frame .
            if self.mdtp.DET_DOWNSAMPLE_ENABLED :
                tDownsample = time.perf_counter()
                currImg = cv2.pyrDown(cframe.mImg, dstsize=(cframe.mImg.shape[1] // 2, cframe.mImg.shape[0] // 2))
                tDownsample = time.perf_counter() - tDownsample
            else :
                currImg = cframe.mImg.copy()

            # Apply mask on currImg .
            # If true is returned, it means that the mask has been updated and applied on currImg. Detection process can't continue .
            # If false is returned, it means that the mask has not been updated. Detection process can continue .
            applied, currImg = self.mMaskManager.applyMask(currFrame=currImg) 
            if applied == False :
                #---------------------------------#
                #      Check previous frame       #
                #---------------------------------#
                if self.mPrevFrame.all() == None :
                    self.mPrevFrame = currImg.copy()
                    return False
                
                #---------------------------------#
                #           Differences           #
                #---------------------------------#
                absDiffImg, posDiffImg, negDiffImg = np.zeros(currImg.shape), np.zeros(currImg.shape), np.zeros(currImg.shape)
                # Absolute difference .
                tAbsDiff = time.perf_counter()
                # cv2.absdiff(currImg, self.mPrevFrame, absDiffImg)
                absDiffImg = cv2.absdiff(currImg, self.mPrevFrame)
                tAbsDiff = time.perf_counter() - tAbsDiff

                # Positive difference .
                tPosDiff = time.perf_counter()
                # cv2.subtract(currImg, self.mPrevFrame, posDiffImg, self.mMaskManager.mCurrentMask)
                posDiffImg = cv2.subtract(currImg, self.mPrevFrame, mask= self.mMaskManager.mCurrentMask)
                
                tPosDiff = time.perf_counter() - tPosDiff

                # Negative difference .
                tNegDiff = time.perf_counter()
                # cv2.subtract(self.mPrevFrame, currImg, negDiffImg, self.mMaskManager.mCurrentMask)
                negDiffImg = cv2.subtract(self.mPrevFrame, currImg, mask=self.mMaskManager.mCurrentMask)
                tNegDiff = tNegDiff - tNegDiff

                # Dilate absolute difference .
                tDilate = time.perf_counter()
                dilationSize = 2 
                structElt = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilationSize + 1, 2*dilationSize + 1 ), (dilationSize, dilationSize))
                # cv2.dilate(absDiffImg, structElt,absDiffImg)
                absDiffImg = cv2.dilate(absDiffImg, structElt)
                tDilate = time.perf_counter() - tDilate

                #---------------------------------------------------------------------------------------------#
                #            Threshold absolute difference/ positive difference/ Negative difference          #
                #---------------------------------------------------------------------------------------------#

                tThreshold = time.perf_counter()
                absDiffBinaryMap = thresholding(absDiffImg, self.mMaskManager.mCurrentMask,10, "MEAN")
                tThreshold = time.perf_counter() - tThreshold
                
                # meanPosDiff, stdevPosDiff, meanNegDiff, stdevNegDiff = 0,0,0,0
                # cv2.meanStdDev(posDiffImg,meanPosDiff,stdevPosDiff, self.mMaskManager.mCurrentMask)
                meanPosDiff,stdevPosDiff = cv2.meanStdDev(posDiffImg,mask= self.mMaskManager.mCurrentMask)
                meanNegDiff, stdevNegDiff = cv2.meanStdDev(negDiffImg, mask= self.mMaskManager.mCurrentMask)
                # cv2.meanStdDev(negDiffImg, meanNegDiff, stdevNegDiff, self.mMaskManager.mCurrentMask)
                posThreshold = stdevPosDiff[0] * 5 + 10
                negThreshold = stdevNegDiff[0] * 5 + 10

                if self.mdtp.DET_DEBUG :
                    posBinaryMap = thresholding(posDiffImg,self.mMaskManager.mCurrentMask, 10, "STDEV")
                    negBinaryMap = thresholding(negDiffImg,self.mMaskManager.mCurrentMask, 10, "STDEV")
                    SaveImg.saveJPEG(currImg,self.mDebugCurrentPath + "/original/frame_" + str(Frame.mFrameNumber))  
                    SaveImg.saveJPEG(posBinaryMap,self.mDebugCurrentPath + "/pos_difference_thresholded/frame_" + str(Frame.mFrameNumber))
                    SaveImg.saveJPEG(negBinaryMap,self.mDebugCurrentPath + "/neg_difference_thresholded/frame_" + str(Frame.mFrameNumber))
                    SaveImg.saveJPEG(absDiffBinaryMap,self.mDebugCurrentPath + "/absolute_difference_thresholded/frame_" + str(Frame.mFrameNumber))
                    SaveImg.saveJPEG(absDiffImg,self.mDebugCurrentPath + "/absolute_difference/frame_" + str(Frame.mFrameNumber))
                    SaveImg.saveJPEG(posDiffImg,self.mDebugCurrentPath + "/pos_difference/frame_" + str(Frame.mFrameNumber))
                    SaveImg.saveJPEG(negDiffImg,self.mDebugCurrentPath + "/neg_difference/frame_" + str(Frame.mFrameNumber))  
                    
                    # cv2.imwrite(f'savedimage{str(Frame.mFrameNumber)}.jpeg', currImg)  
                    Frame.mFrameNumber += 1

                # Current frame is stored as the previous frame .
                self.mPrevFrame = currImg.copy()
                tStep1 = time.perf_counter() - tStep1

                ###########################################################################
                #                       STEP 2 : FIND LOCAL EVENTS                        #
                ###########################################################################
                
                # SUMMARY :
                # Loop binarized absolute difference image.
                # For each white pixel, define a Region of interest (ROI) of 10x10 centered in this pixel.
                # Create a new Local Event initialized with this first ROI or attach this ROI to an existing Local Event.
                # Loop the ROI in the binarized absolute difference image to store position of white pixels.
                # Loop the ROI in the positive difference image to store positions of white pixels.
                # Loop the ROI in the negative difference image to store positions of white pixels.
                # Once the list of Local Event has been completed :
                # Analyze each local event in order to check that pixels can be clearly split in two groups (negative, positive).
            else :
                # Current frame is stored as the previous frame .
                self.mPrevFrame = currImg.copy()
            return np.array(currImg, dtype= np.uint8)

        pass

    def saveDetectionInfos(self, p, nbFramesAround):

        # Save GE map .
        if self.mdtp.temporal.DET_SAVE_GEMAP :
            SaveImg.saveBMP(img= self.mGeToSave.getMapEvent(), name= p + "GeMap")
            self.debugFiles = np.append(self.debugFiles, "GeMap.bmp")
        # Save dir map .
        if self.mdtp.temporal.DET_SAVE_DIRMAP :
            SaveImg.saveBMP(img=self.mGeToSave.getDirMap(), name= p+"DirMap")

        # # Save infos.
        # if self.mdtp.temporal.DET_SAVE_GE_INFOS:
        #     infFilePath = p + "GeInfos.txt"
        #     with open(infFilePath, 'w') as infFile:
        #         infFile.write(" * AGE              : {}\n".format(self.mGeToSave.getAge()))
        #         infFile.write(" * AGE LAST ELEM    : {}\n".format(self.mGeToSave.getAgeLastElem()))
        #         infFile.write(" * LINEAR STATE     : {}\n".format(self.mGeToSave.getLinearStatus()))
        #         infFile.write(" * BAD POS          : {}\n".format(self.mGeToSave.getBadPos()))
        #         infFile.write(" * GOOD POS         : {}\n".format(self.mGeToSave.getGoodPos()))
        #         infFile.write(" * NUM FIRST FRAME  : {}\n".format(self.mGeToSave.getNumFirstFrame()))
        #         infFile.write(" * NUM LAST FRAME   : {}\n".format(self.mGeToSave.getNumLastFrame()))

        #         d = sqrt(pow(self.mGeToSave.mainPts[-1].x - self.mGeToSave.mainPts[0].x, 2.0) + pow(self.mGeToSave.mainPts[-1].y - self.mGeToSave.mainPts[0].y, 2.0))
        #         infFile.write("\n * Distance between first and last  : {}\n".format(d))

        #         infFile.write("\n * MainPoints position : \n")
        #         for i in range(len(self.mGeToSave.mainPts)):
        #             infFile.write("    ({}, {})\n".format(self.mGeToSave.mainPts[i].x, self.mGeToSave.mainPts[i].y))

        #         infFile.write("\n * MainPoints details : \n")
        #         for i in range(len(self.mGeToSave.listA)):
        #             infFile.write("    A({}, {}) ----> ".format(self.mGeToSave.listA[i].x, self.mGeToSave.listA[i].y))
        #             infFile.write("    B({}, {}) ----> ".format(self.mGeToSave.listB[i].x, self.mGeToSave.listB[i].y))
        #             infFile.write("    C({}, {})\n".format(self.mGeToSave.listC[i].x, self.mGeToSave.listC[i].y))
        #             infFile.write("    u({}, {})       ".format(self.mGeToSave.listu[i].x, self.mGeToSave.listu[i].y))
        #             infFile.write("    v({}, {})\n".format(self.mGeToSave.listv[i].x, self.mGeToSave.listv[i].y))
        #             infFile.write("    Angle rad between BA' / BC = {}\n".format(self.mGeToSave.listRad[i]))
        #             infFile.write("    Angle between BA' / BC = {}\n".format(self.mGeToSave.listAngle[i]))

        #             if self.mGeToSave.mainPtsValidity[i]:
        #                 infFile.write("    NEW POSITION ACCEPTED\n\n")
        #             else:
        #                 infFile.write("    NEW POSITION REFUSED\n\n")

        # Save positions .
        if self.mdtp.temporal.DET_SAVE_POS:
            posFile = open(p + "positions.txt", "w")

            # Number of the first frame associated to the event.
            numFirstFrame = -1

            for itLe in self.mGeToSave.LEList:

                if numFirstFrame == -1:
                    numFirstFrame = itLe.getNumFrame()

                pos:Point = itLe.getMassCenter()

                positionY = 0
                if self.mdtp.DET_DOWNSAMPLE_ENABLED:
                    pos *= 2
                    positionY = self.mPrevFrame.rows * 2 - pos.y
                else:
                    positionY = self.mPrevFrame.rows - pos.y

                # NUM_FRAME    POSITIONX     POSITIONY (inversed)
                line = str(itLe.getNumFrame() - numFirstFrame + nbFramesAround) + "               (" + str(pos.x) + ";" + str(positionY) + ")                 " + TimeDate.getIsoExtendedFormatDate(itLe.mFrameAcqDate) + "\n"
                posFile.write(line)

            posFile.close()

    def resetDetection(self, loadNewDataSet):

        self.mListGlobalEvents = np.array([], dtype= GlobalEvent)

        # Clear list of files to send by mail.
        self.debugFiles = np.array([], dtype=str)
        self.mSubdivisionStatus = False
        self.mPrevThresholdedMap = None
        self.mPrevFrame = None

        if self.mdtp.DET_DEBUG and loadNewDataSet:
            self.mDataSetCounter += 1
            self.createDebugDirectories(False)

    def resetMask(self):
        self.mMaskManager.resetMask()

    def createDebugDirectories(self, cleanDebugDirectory):
        self.mDebugCurrentPath = self.mdtp.DET_DEBUG_PATH + "debug_" + str(self.mDataSetCounter) + "/"

        if cleanDebugDirectory :
            p0 = pathlib.Path(self.mdtp.DET_DEBUG_PATH)
            if p0.exists():
                # pathlib.Path.unlink(p0)
                pass
            else :
                pathlib.Path.mkdir(p0, parents=True)

            p1 = pathlib.Path(self.mDebugCurrentPath)
            if not p1.exists():
                pathlib.Path.mkdir(p1)      

            debugSubDir = ["original", "absolute_difference", "event_map_initial", "event_map_filtered", "absolute_difference_thresholded", "neg_difference_thresholded", "pos_difference_thresholded", "neg_difference", "pos_difference"]  
            for sub_dir in debugSubDir:
                path = pathlib.Path(self.mDebugCurrentPath + sub_dir)

                if not path.exists():
                    pathlib.Path.mkdir(path)

    def getColorInEventMap(self, eventMap, roiCenter:Point):
        # ROI in the eventMap.
        roi = eventMap[
            roiCenter.y - self.mRoiSize[1] // 2 : roiCenter.y + self.mRoiSize[1] // 2,
            roiCenter.x - self.mRoiSize[0] // 2 : roiCenter.x + self.mRoiSize[0] // 2
        ].copy()

        listColor = []
        exist = False

        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                bgrPixel = tuple(roi[i, j])
                
                if bgrPixel[0] != 0 or bgrPixel[1] != 0 or bgrPixel[2] != 0:
                    for k in range(len(listColor)):
                        if bgrPixel == listColor[k]:
                            exist = True
                    
                    if not exist:
                        listColor.append(bgrPixel)
                    
                    exist = False
        
        return np.array(listColor)

    def colorRoiInBlack(self, p:Point, h, w, region):
        posX = p.y - w
        posY = p.x - h

        if p.y - w < 0:
            w = p.y + w // 2
            posX = 0
        elif p.y + w // 2 > region.shape[1]:
            w = region.shape[1] - p.y + w // 2

        if p.x - h < 0:
            h = p.x + h // 2
            posY = 0
        elif p.x + h // 2 > region.shape[0]:
            h = region.shape[0] - p.x + h // 2

        # Color ROI in black in the current region.
        roiBlackRegion = np.zeros((h, w), dtype=np.uint8)
        region[posY:posY+h, posX:posX+w] = roiBlackRegion
        
        return region

    def analyseRegion(self, subdivision, absDiffBinaryMap, eventMap, posDiff, posDiffThreshold, negDiff,
                      negDiffThreshold, listLE, subdivisionPos, maxNbLE, numFrame, msg, cFrameDate):
        situation = 0
        nbCreatedLE = 0
        nbRoiAttachedToLE = 0
        nbNoCreatedLE = 0
        nbROI = 0
        nbRoiNotAnalysed = 0
        roicounter = 0

        # Loop pixel's subdivision.
        for i in range(subdivision.shape[0]):
            for j in range(subdivision.shape[1]):
                # Pixel is white 
                if subdivision[i,j] > 0 :
                    # Check if we are not out of frame range when a ROI is defined at the current pixel location .
                    if (subdivisionPos.y + j - self.mRoiSize[1]//2 > 0) and (subdivisionPos.y + j + self.mRoiSize[1]//2 < absDiffBinaryMap.shape[1]) and (subdivisionPos.x + i - self.mRoiSize[0]//2 > 0) and (subdivisionPos.x + i + self.mRoiSize[0]//2 < absDiffBinaryMap.shape[0]):
                        msg = msg + "Analyse ROI (" + str(subdivisionPos.x + i) + ";" + str(subdivisionPos.y + j) + ")\n"
                        nbROI += 1
                        roicounter += 1
                        # Get Colors in eventMap at the current ROI location .
                        listColorInRoi = self.getColorInEventMap(eventMap, Point(subdivisionPos.x + i, subdivisionPos.y + j))
                        if len(listColorInRoi) == 0:
                            situation = 0  # black color = create a new local event
                        elif len(listColorInRoi) == 1:
                            situation = 1  # one color = add the current roi to an existing local event
                        else:
                            situation = 2  # several colors = make a decision


                        if situation == 0 :
                            if listLE.size() < maxNbLE :
                                msg = msg + "->CREATE New Local EVENT\n" \
                                    + "  - Initial position : (" \
                                    + str(subdivisionPos.x + i) + ";" + str(subdivisionPos.y + j) + ")\n" \
                                    + "  - Color : (" + str(self.mListColors[listLE.size()][0]) + ";" \
                                    + str(self.mListColors[listLE.size()][1]) + ";" \
                                    + str(self.mListColors[listLE.size()][2]) + ")\n"
                                
                                # Create new localEvent object .
                                newLocalEvent = LocalEvent(color=self.mListColors[-1], roiPos=Point(x=subdivisionPos.x+i, y=subdivisionPos.y+j), frameHeight=absDiffBinaryMap.shape[0], frameWidth=absDiffBinaryMap.shape[1], roiSize=self.mRoiSize)
                                
                                # Extract white pixels in ROI .
                                whitePixAbsDiff, whitePixPosDiff, whitePixNegDiff = [], [], []
                                roiAbsDiff = absDiffBinaryMap[subdivisionPos.x + i - self.mRoiSize[0]//2 : subdivisionPos.x + i + self.mRoiSize[0]//2,
                                                          subdivisionPos.y + j - self.mRoiSize[1]//2 : subdivisionPos.y + j + self.mRoiSize[1]//2].copy()
                                roiPosDiff = posDiff[subdivisionPos.x + i - self.mRoiSize[0]//2 : subdivisionPos.x + i + self.mRoiSize[0]//2,
                                                          subdivisionPos.y + j - self.mRoiSize[1]//2 : subdivisionPos.y + j + self.mRoiSize[1]//2].copy()
                                roiNegDiff = negDiff[subdivisionPos.x + i - self.mRoiSize[0]//2 : subdivisionPos.x + i + self.mRoiSize[0] // 2,
                                                     subdivisionPos.y + j - self.mRoiSize[1] // 2 : subdivisionPos.y + j + self.mRoiSize[1] // 2].copy()
                                
                                if roiPosDiff.type() == np.uint16 and roiNegDiff.type() == np.uint16 :
                                    for a in range(roiAbsDiff.shape[0]):
                                        for b in range(roiAbsDiff.shape[1]):
                                            if roiAbsDiff[a,b] > 0 :
                                                whitePixAbsDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                            if roiPosDiff[a,b] > posDiffThreshold : 
                                                whitePixPosDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                            if roiNegDiff[a,b] > negDiffThreshold :
                                                whitePixNegDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                
                                elif roiPosDiff.type() == np.uint8 and roiNegDiff.type() == np.uint8 :
                                    for a in range(roiAbsDiff.shape[0]):
                                        for b in range(roiAbsDiff.shape[1]):
                                            if roiAbsDiff[a,b] > 0 :
                                                whitePixAbsDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                            if roiPosDiff[a,b] > posDiffThreshold : 
                                                whitePixPosDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                            if roiNegDiff[a,b] > negDiffThreshold :
                                                whitePixNegDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))

                                msg = msg + "Number white pix in abs diff : " + str(len(whitePixAbsDiff)) + "\n"
                                msg = msg + "Number white pix in pos diff : " + str(len(whitePixPosDiff)) + "\n"
                                msg = msg + "Number white pix in neg diff : " + str(len(whitePixNegDiff)) + "\n"

                                newLocalEvent.addAbs(whitePixAbsDiff)
                                newLocalEvent.addPos(whitePixPosDiff)
                                newLocalEvent.addNeg(whitePixNegDiff)

                                # Update center of mass .
                                newLocalEvent.computeMassCenter()
                                msg = msg + "  - Center of mass abs pixels : (" + str(newLocalEvent.getMassCenter().x) + ";" + str(newLocalEvent.getMassCenter().y) + ")\n"

                                # Save the frame number where the local event has been created .
                                newLocalEvent.setNumFrame(numFrame)
                                # Save acquisition date of the frame .
                                newLocalEvent.mFrameAcqDate = cFrameDate
                                # Add the LE in the list of localEvent .
                                listLE.append(newLocalEvent)
                                # Update eventMap with the color of the new localEvent .
                                roi = np.full((self.mRoiSize[0], self.mRoiSize[1]),self.mListColors[-1],dtype=np.uint8)
                                eventMap[subdivisionPos.x + i - self.mRoiSize[0]//2:subdivisionPos.x + i + self.mRoiSize[0]//2, subdivisionPos.y + j - self.mRoiSize[1] // 2 : subdivisionPos.y + j + self.mRoiSize[1] // 2] = roi.copy()
                                # Color the roi in black in the current region .
                                subdivision = self.colorRoiInBlack(Point(i,j), self.mRoiSize[0], self.mRoiSize[1],subdivision)
                                absDiffBinaryMap = self.colorRoiInBlack(Point(subdivisionPos.x + i,subdivisionPos + j), self.mRoiSize[0], self.mRoiSize[1],absDiffBinaryMap)
                                posDiff = self.colorRoiInBlack(Point(subdivisionPos.x + i,subdivisionPos + j), self.mRoiSize[0], self.mRoiSize[1],posDiff)
                                negDiff = self.colorRoiInBlack(Point(subdivisionPos.x + i,subdivisionPos + j), self.mRoiSize[0], self.mRoiSize[1],negDiff)

                                nbCreatedLE += 1
                            else :
                                nbNoCreatedLE += 1

                        elif situation == 1 :
                            index = 0
                            for le in listLE :
                                # Try to find a local event which has the same color .
                                if le.getColor() == listColorInRoi[0]:
                                    msg = msg + "->Attach ROI (" + str(subdivisionPos.x + i) + ";" + str(subdivisionPos.y + j) + ") with LE " + str(index) + "\n"
                                    # Extract white pixels in ROI .
                                    whitePixAbsDiff, whitePixPosDiff, whitePixNegDiff = [], [], []
                                    roiAbsDiff = absDiffBinaryMap[subdivisionPos.x + i - self.mRoiSize[0]//2 : subdivisionPos.x + i + self.mRoiSize[0]//2,
                                                          subdivisionPos.y + j - self.mRoiSize[1]//2 : subdivisionPos.y + j + self.mRoiSize[1]//2].copy()
                                    roiPosDiff = posDiff[subdivisionPos.x + i - self.mRoiSize[0]//2 : subdivisionPos.x + i + self.mRoiSize[0]//2,
                                                          subdivisionPos.y + j - self.mRoiSize[1]//2 : subdivisionPos.y + j + self.mRoiSize[1]//2].copy()
                                    roiNegDiff = negDiff[subdivisionPos.x + i - self.mRoiSize[0]//2 : subdivisionPos.x + i + self.mRoiSize[0] // 2,
                                                     subdivisionPos.y + j - self.mRoiSize[1] // 2 : subdivisionPos.y + j + self.mRoiSize[1] // 2].copy()
                                    
                                    if roiPosDiff.type() == np.uint16 and roiNegDiff.type() == np.uint16 :
                                        for a in range(roiAbsDiff.shape[0]):
                                            for b in range(roiAbsDiff.shape[1]):
                                                if roiAbsDiff[a,b] > 0 :
                                                    whitePixAbsDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                                if roiPosDiff[a,b] > posDiffThreshold : 
                                                    whitePixPosDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                                if roiNegDiff[a,b] > negDiffThreshold :
                                                    whitePixNegDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                
                                    elif roiPosDiff.type() == np.uint8 and roiNegDiff.type() == np.uint8 :
                                        for a in range(roiAbsDiff.shape[0]):
                                            for b in range(roiAbsDiff.shape[1]):
                                                if roiAbsDiff[a,b] > 0 :
                                                    whitePixAbsDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                                if roiPosDiff[a,b] > posDiffThreshold : 
                                                    whitePixPosDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                                if roiNegDiff[a,b] > negDiffThreshold :
                                                    whitePixNegDiff.append(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2 + a,y=subdivisionPos.y + j - self.mRoiSize[1]//2 + b))
                                    msg = msg + "Number white pix in abs diff : " + str(len(whitePixAbsDiff)) + "\n"
                                    msg = msg + "Number white pix in pos diff : " + str(len(whitePixPosDiff)) + "\n"
                                    msg = msg + "Number white pix in neg diff : " + str(len(whitePixNegDiff)) + "\n"

                                    le.addAbs(whitePixAbsDiff)
                                    le.addPos(whitePixPosDiff)
                                    le.addNeg(whitePixNegDiff)

                                    # Add the current roi .
                                    le.mLeRoiList = np.append(le.mLeRoiList, Point(x=subdivisionPos.x + i, y= subdivisionPos.y + j))
                                    # Set the Local event's map .
                                    le.setMap(Point(x=subdivisionPos.x + i - self.mRoiSize[0]//2, y= subdivisionPos.y + j - self.mRoiSize[1]//2), self.mRoiSize)
                                    # Update center of mass .
                                    le.computeMassCenter()
                                    msg = msg + "  - Update Center of mass abs pixels of LE " + str(index) + " : (" + str(le.getMassCenter().x) + ";" + str(le.getMassCenter().y) + ")\n"
                                    # Update eventMap with the color of the new localEvent .
                                    roi = np.full((self.mRoiSize[0], self.mRoiSize[1]),self.mListColors[0],dtype=np.uint8)
                                    eventMap[subdivisionPos.x + i - self.mRoiSize[0]//2:subdivisionPos.x + i + self.mRoiSize[0]//2, subdivisionPos.y + j - self.mRoiSize[1] // 2 : subdivisionPos.y + j + self.mRoiSize[1] // 2] = roi.copy()

                                    # Color roi in black in thresholded frame .
                                    roiBlack = np.full((self.mRoiSize[0], self.mRoiSize[1]),self.mListColors[0],dtype=np.uint8)
                                    absDiffBinaryMap[subdivisionPos.x + i - self.mRoiSize[0]//2:subdivisionPos.x + i + self.mRoiSize[0]//2, subdivisionPos.y + j - self.mRoiSize[1] // 2 : subdivisionPos.y + j + self.mRoiSize[1] // 2] = roiBlack.copy()
                                    # Color the roi in black in the current region .
                                    subdivision = self.colorRoiInBlack(Point(i,j), self.mRoiSize[0], self.mRoiSize[1],subdivision)
                                    absDiffBinaryMap = self.colorRoiInBlack(Point(subdivisionPos.x + i,subdivisionPos + j), self.mRoiSize[0], self.mRoiSize[1],absDiffBinaryMap)
                                    posDiff = self.colorRoiInBlack(Point(subdivisionPos.x + i,subdivisionPos + j), self.mRoiSize[0], self.mRoiSize[1],posDiff)
                                    negDiff = self.colorRoiInBlack(Point(subdivisionPos.x + i,subdivisionPos + j), self.mRoiSize[0], self.mRoiSize[1],negDiff)

                                    nbRoiAttachedToLE += 1
                                    break
                                index += 1

                        elif situation == 2 :
                            nbRoiNotAnalysed += 1
                            # LoEv = False
                            # colorFound = False
        msg = msg + "--> RESUME REGION ANALYSE : \n" \
        + "Number of analysed ROI : " + str(nbROI) + "\n" \
        + "Number of not analysed ROI : " + str(nbRoiNotAnalysed) + "\n" \
        + "Number of new LE : " + str(nbCreatedLE) + "\n" \
        + "Number of updated LE :" + str(nbRoiAttachedToLE) + "\n"
        return listLE, subdivision, eventMap, absDiffBinaryMap, posDiff, negDiff, msg

    # def selectThreshold(self, i):
    #     pass
    def getEventFirstFrameNb(self):
        return self.mGeToSave.geFirstFrameNum

    def getEventDate(self):
        return self.mGeToSave.date

    def getEventLastFrameNb(self):
        return self.mGeToSave.geLastFrameNum

    def getDebugFiles(self):
        return self.debugFiles