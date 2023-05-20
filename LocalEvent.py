import cv2
import numpy as np
from math import sqrt, pow, acos
from Circle import *
from SParam import detectionParam


class LocalEvent:
    def __init__(self, color, roiPos:Point, frameHeight, frameWidth, roiSize):

        # should be private .
        self.mLeColor = color      # Color attribute of the local event .    
        self.mLeMap  = np.zeros((frameHeight,frameWidth), dtype=np.uint8) # ROI map .    
        self.mLeMassCenter:Point = None     # Local Event mass center .
        self.mLeNumFrame = 0       # Associated frame .

        self.mPosMassCenter:Point = Point(0,0)
        self.mNegMassCenter:Point = Point(0,0)
        self.mPosRadius = 0.5
        self.mNegRadius = 0.5
        self.mPosCluster = False
        self.mNegCluster = False
        self.mergedFlag = False
        self.uNegToPos:Point = Point(0,0)
        self.index = 0

        # should be public .
        self.mLeRoiList = []    # Contains position of regions of interest which composes a local event .
        self.mAbsPos = []
        self.mPosPos = []
        self.mNegPos = []
        self.mFrameHeight = frameHeight
        self.mFrameWidth = frameWidth
        self.mFrameAcqDate = None

        # Save position of the first ROI .
        self.mLeRoiList.append(roiPos)

        # Add the first ROI to the LE map .
        self.roi = np.full((roiSize[0], roiSize[1]), 255, dtype=np.uint8)
        x, y = roiPos.x, roiPos.y
        self.mLeMap[x - roiSize[0] // 2:x + roiSize[0] // 2, y - roiSize[1] // 2:y + roiSize[1] // 2] = self.roi

    def computeMassCenter(self):
        total_x = sum(point.x for point in self.mAbsPos)
        total_y = sum(point.y for point in self.mAbsPos)
        length = len(self.mAbsPos)
        if length == 0:
            self.mLeMassCenter = Point(0, 0)
        else:
            self.mLeMassCenter = Point(int(total_x / length), int(total_y / length))

    def setMap(self, roiPos:Point, roiSize):
        # Add the first ROI to the LE map .
        roiH, roiW = roiSize[0], roiSize[1]
        self.roi = np.full((roiH, roiW), 255, dtype=np.uint8)
        x, y = roiPos.x, roiPos.y
        self.mLeMap[x:x+roiH, y:y+roiW]
    
    def addAbs(self, points):
        self.mAbsPos = self.mAbsPos + points

    def addPos(self, points):
        self.mPosPos = self.mPosPos + points
        if len(self.mPosPos) != 0 :
            self.mPosCluster = True

    def addNeg(self, points):
        self.mNegPos = self.mNegPos + points
        if len(self.mNegPos) != 0 :
            self.mNegCluster = True

    def createPosNegAbsMap(self):
        mapy = np.zeros((self.mFrameHeight, self.mFrameWidth), dtype=np.uint8)
        
        for absPoint in self.mAbsPos :
            # Set the RGB channels into a balck color .
            mapy[absPoint.x,absPoint.y] = np.array([255,255,255])
        lengthPosPos = len(self.mPosPos)
        if lengthPosPos != 0 :
            xPos, yPos = 0.0, 0.0
            for posPoint in self.mPosPos :
                mapy[posPoint.x, posPoint.y] = np.array([0,255,0])
                xPos += posPoint.x
                yPos += posPoint.y
            xPos = xPos // lengthPosPos
            yPos = yPos // lengthPosPos

            self.mPosMassCenter = Point(x=int(xPos), y=int(yPos))

            # search radius .
            posRadius = 0.0
            for posPoint in self.mPosPos :
                radius = sqrt(pow(posPoint.x-self.mPosMassCenter.x, 2)+pow(posPoint.y-self.mPosMassCenter.y, 2))  
                if radius > posRadius :
                    posRadius = radius
            if self.mPosMassCenter.x > 0 and self.mPosMassCenter.y > 0 :
                # Draw a green circle on the image around the mass center with the calculated radius.
                cv2.circle(mapy, (int(self.mPosMassCenter.x), int(self.mPosMassCenter.y)), int(posRadius), (0, 255, 0), thickness=-1)
        lengthNegPos = len(self.mNegPos)
        if lengthNegPos != 0 :
            xNeg, yNeg = 0.0, 0.0
            for negPoint in self.mNegPos :
                if mapy[negPoint.x, negPoint.y] == [0,255,0]:
                    mapy[negPoint.x, negPoint.y] = np.array([255,0,0])
                else :
                    mapy[negPoint.x, negPoint.y] = np.array([0,0,255])

                xNeg += negPoint.x
                yNeg += negPoint.y
            xNeg = xNeg // lengthNegPos
            yNeg = yNeg // lengthNegPos

            self.mNegMassCenter = Point(x=int(xNeg), y=int(yNeg))

            # search radius .
            negRadius = 0.0
            for negPoint in self.mNegPos :
                radius = sqrt(pow(negPoint.x-self.mNegMassCenter.x, 2)+pow(negPoint.y-self.mNegMassCenter.y,2))  
                if radius > self.mNegRadius :
                    negRadius = radius
            if self.mPosMassCenter.x > 0 and self.mPosMassCenter.y > 0 :
                # Draw a green circle on the image around the mass center with the calculated radius.
                cv2.circle(map, (int(self.mNegMassCenter.x), int(self.mNegMassCenter.y)), int(negRadius), (0,0,255), thickness=-1)
        return mapy

    def localEventIsValid(self):
        posCluster, negCluster = False, False

        # Positive cluster .
        if len(self.mPosPos) != 0 :
            self.computePosMassCenter()
            self.computePosRadius()
            if self.mPosRadius > 0.0 :
                posCluster = True

        # Negative cluster .
        if len(self.mNegPos) != 0 :
            self.computeNegMassCenter()
            self.computeNegRadius()
            if self.mNegRadius > 0.0 :
                negCluster = True

        # Check intersection between clusters .
        if negCluster and posCluster :
            # Vector from neg cluster to pos cluster .
            self.uNegToPos = Point(x=self.mPosMassCenter.x-self.mNegMassCenter.x, y=self.mPosMassCenter.y-self.mNegMassCenter.y)
            # Create two circles around the positive and negative clusters with the calculated radiuses .
            posCircle = Circle(self.mPosMassCenter, self.mPosRadius)
            negCircle = Circle(self.mNegMassCenter, self.mNegRadius)
            dtp = detectionParam()
            # Check if there is an intersection between the two circles .
            res, surfaceCircle1 , surfaceCircle2 , intersectedSurface = posCircle.computeDiskSurfaceIntersection(negCircle, dtp.DET_DEBUG, dtp.DET_DEBUG_PATH)
            if not res :
                return True
            elif surfaceCircle1 != 0 and intersectedSurface !=0 and surfaceCircle2 != 0 :
                # One of the two circles is intersected more than 50% of its surface .
                if (intersectedSurface * 100)/surfaceCircle1 > 50 or (intersectedSurface * 100)/surfaceCircle2 > 50 :
                    return False # LE is not valid .

                else :
                    return True # LE is valid .
            else:
                return False # LE is not valid .
            
        return True

    def mergeWithAnotherLE(self, LE:"LocalEvent"):
        print("merging")
        self.mLeRoiList = self.mLeRoiList + LE.mLeRoiList
        self.completeGapWithRoi(self.mLeMassCenter, LE.getMassCenter())
        self.mAbsPos = self.mAbsPos + LE.mAbsPos
        self.mPosPos = self.mPosPos + LE.mPosPos
        self.mNegPos = self.mNegPos + LE.mNegPos
        self.computeMassCenter()
        temp = self.mLeMap + LE.getMap()
        self.mLeMap = temp.copy()
        if len(self.mPosPos) !=0 : self.mPosCluster = True
        if len(self.mNegPos) !=0 : self.mNegCluster = True

    def completeGapWithRoi(self, p1:Point, p2:Point):
        roi = np.full((10,10),255,dtype=np.uint8)
        dist = sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2))
        part = dist / 10.0
        if int(part) != 0 :
            p3 = Point(x=p1.x, y=p2.y)
            dist1 = sqrt(pow(p1.x-p3.x,2)+pow(p1.y-p3.y,2))
            dist2 = sqrt(pow(p2.x-p3.x,2)+pow(p2.y-p3.y,2))
            part1 = dist1/part
            part2 = dist2/part
            for i in range(0,int(part)):
                p = Point(x=int(p3.x+i*part2), y=int(p1.y+i*part1))
                if p.x-5 > 0 and p.x+5 < self.mLeMap.shape[0] and p.y-5 > 0 and p.y+5 < self.mLeMap.shape[1] :
                    self.mLeMap[p.x-5:p.x+5, p.y-5:p.y+5] = roi.copy()

    def computePosMassCenter(self):
        total_x = sum(point.x for point in self.mPosPos)
        total_y = sum(point.y for point in self.mPosPos)
        length = len(self.mPosPos)
        if length == 0:
            self.mPosMassCenter = Point(0, 0)
        else:
            self.mPosMassCenter = Point(int(total_x / length), int(total_y / length))

    def computeNegMassCenter(self):
        total_x = sum(point.x for point in self.mNegPos)
        total_y = sum(point.y for point in self.mNegPos)
        length = len(self.mNegPos)
        if length == 0:
            self.mNegMassCenter = Point(0, 0)
        else:
            self.mNegMassCenter = Point(int(total_x / length), int(total_y / length))
    
    def computePosRadius(self):
        # Search for radius .
        for point in self.mPosPos :
            radius = sqrt(pow(point.x- self.mPosMassCenter.x, 2)+pow(point.y-self.mPosMassCenter.y,2))
            if radius > self.mPosRadius :
                self.mPosRadius = radius

    def computeNegRadius(self):
        # Search for radius .
        for point in self.mNegPos :
            radius = sqrt(pow(point.x- self.mNegMassCenter.x, 2)+pow(point.y-self.mNegMassCenter.y, 2))
            if radius > self.mNegRadius :
                self.mNegRadius = radius

    def getColor(self):
        return self.mLeColor

    def getMap(self):
        return self.mLeMap

    def getMassCenter(self):
        return self.mLeMassCenter
    
    def getPosMassCenter(self):
        return self.mPosMassCenter

    def getNegMassCenter(self):
        return self.mNegMassCenter
    
    def getPosRadius(self):
        return self.mPosRadius
    
    def getNegRadius(self):
        return self.mNegRadius

    def getNumFrame(self):
        return self.mLeNumFrame
    
    def setNumFrame(self, numFrame):
        self.mLeNumFrame = numFrame
    
    def getPosCluster(self):
        return self.mPosCluster

    def getNegCluster(self):
        return self.mNegCluster
    
    def getPosClusterStatus(self):
        return self.mPosCluster

    def getNegClusterStatus(self):
        return self.mNegCluster

    def getMergedStatus(self):
        return self.mergedFlag

    def setMergedStatus(self, flag):
        self.mergedFlag = flag

    def getMergedFlag(self):
        return self.mergedFlag

    def getLeDir(self):
        return self.uNegToPos

    def getLeIndex(self):
        return self.index

    def setLeIndex(self, i):
        self.index = i