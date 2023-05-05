import cv2
import numpy as np
from LocalEvent import *
from Frame import Frame
from Types import Point

class GlobalEvent :
    def __init__(self, frameDate, frameNum, frameHeight, frameWidth, color):
        # private vars :
        self.geAge = 0
        self.geAgeLastLE = 0
        self.geDate = frameDate
        self.geMap = np.full((frameHeight,frameWidth),0,dtype=np.uint8)
        self.geFirstFrameNum = frameNum
        self.geLastFrameNum =frameNum
        self.geDirMap = np.full((frameHeight, frameWidth,3), 0,dtype=np.uint8)
        self.geShifting = 0
        self.newLeAdded = False
        self.geLinear = True
        self.geBadPoint = 0
        self.geGoodPoint = 0
        self.geColor = color
        self.geMapColor = np.full((frameHeight, frameWidth, 3), 0,dtype=np.uint8)
        # public vars :
        self.LEList = np.array([], dtype=LocalEvent)
        self.ptsValidity = np.array([], dtype=bool)
        self.distBtwPts = 0
        self.distBtwMainPts = 0
        self.mainPts = np.array([], dtype=Point)
        self.pts = np.array([], dtype=Point)
        self.leDir:Point = None
        self.geDir:Point = None
        self.listA = np.array([], dtype=Point) 
        self.listB = np.array([], dtype=Point)
        self.listC = np.array([], dtype=Point)
        self.listu = np.array([], dtype=Point) # List of vectors from first main point to last main point position .
        self.listv = np.array([], dtype=Point) # List of vectors from last main point to current LE position .
        self.listAngle = np.array([], dtype=float)
        self.listRad = np.array([], dtype=float)
        self.mainPtsValidity = np.array([], dtype=bool)
        self.clusterNegPos = np.array([], dtype=bool)

    def addLE(self, le:LocalEvent) :
        # Get LE position .
        center = Point(x=le.getMassCenter().x, y=le.getMassCenter().y)

        # Indicates if the LE in input can be added to the global event .
        addLeDecision = True

        # First LE's position become a main point .
        if self.pts.size == 0 :
            self.mainPts = np.append(self.mainPts, center)
            self.geGoodPoint += 1
            self.ptsValidity = np.append(self.ptsValidity, True)

        # If the current LE is at least the second .
        elif self.pts.size > 0:
            if self.listv.size > 1 :
                scalar = le.getLeDir().x * self.listv[-1].x + le.getLeDir().y * self.listv[-1].y
                self.leDir = le.getLeDir()
                if scalar <= 0.0 :
                    self.clusterNegPos = np.append(self.clusterNegPos, False)
                else :
                    self.clusterNegPos = np.append(self.clusterNegPos, True)
            # Check global event direction each 3 LEs .
            if (self.pts.size +1)%3 == 0 :
                # If there is at least 2 main points .
                if self.mainPts.size >=2 :
                    # Get the first main point .
                    A = self.mainPts[0]
                    self.listA = np.append(self.listA, A)
                    # Get the last main point .
                    B = self.mainPts[-1]
                    self.listB = np.append(self.listB, B)
                    # Get current LE position .
                    C = center
                    self.listC = np.append(self.listC, C)
                    # Vector from first main point to last main point .
                    u= Point(x= B.x - A.x, y= B.y - A.y)
                    self.listu = np.append(self.listu, u)

                    # Vector from last main point to current LE position .
                    v = Point(x = C.x - B.x, y= C.y - B.y)
                    self.listv = np.append(self.listv, v)
                    self.geDir = v

                    # Same mainPts position : No displacement .
                    if (v.x == 0 and v.y == 0) or (u.x == 0 and u.y == 0) :
                        self.listRad = np.append(self.listRad, 0)
                        self.listAngle = np.append(self.listAngle, 0)
                        self.mainPtsValidity = np.append(self.mainPtsValidity, False)
                        addLeDecision = False
                        self.ptsValidity = np.append(self.ptsValidity, False)
                        
                        # self.geBadPoint = 0
                        # self.geGoodPoint +=1
                        # self.mainPts = np.append(self.mainPts, center)
                        # self.ptsValidity = np.append(self.ptsValidity, True)
                        # self.mainPtsValidity = np.append(self.mainPtsValidity, True)
                        # cv2.circle(self.geDirMap, (center.y, center.x), 5, (255, 0, 0), 1, cv2.LINE_AA, 0)
                    else :
                        # Birds filter 
                        scalar = le.getLeDir().x * v.x + le.getLeDir().y * v.y
                        if scalar <= 0.0 :
                            self.clusterNegPos = np.append(self.clusterNegPos, False)
                        else :
                            self.clusterNegPos = np.append(self.clusterNegPos, True)

                        # Norm vector u .
                        normU = sqrt(pow(u.x,2) + pow(u.y,2))
                        # Norm vector v .
                        normV = sqrt(pow(v.x,2)+pow(v.y,2))
                        # Compute angle between u and v .
                        thetaRad = round((u.x*v.x+u.y*v.y)/(normU*normV),8)
                        self.listRad = np.append(self.listRad, thetaRad)
                        thetaDeg = round((180 * acos(thetaRad))/pi, 8)
                        self.listAngle = np.append(self.listAngle, thetaDeg)
                        
                        if thetaDeg > 40.0 or thetaDeg < -40.0 :
                            self.geBadPoint += 1
                            if self.geBadPoint == 3 :
                                self.geLinear = False
                            addLeDecision = False
                            self.ptsValidity = np.append(self.ptsValidity, False)
                            self.mainPtsValidity = np.append(self.mainPtsValidity, False)
                            cv2.circle(self.geDirMap, (center.y, center.x), 5, (0, 0, 255), 1, cv2.LINE_AA, 0)
                        else :
                            self.geBadPoint = 0
                            self.geGoodPoint += 1
                            self.mainPts = np.append(self.mainPts, center)
                            self.ptsValidity = np.append(self.ptsValidity, True)
                            self.mainPtsValidity = np.append(self.mainPtsValidity, True)
                            cv2.circle(self.geDirMap, (center.y, center.x), 5, (255, 255, 255), 1, cv2.LINE_AA, 0)
                else :
                    # Create new main point .
                    self.mainPts = np.append(self.mainPts, center)
                    self.geGoodPoint += 1
                    self.ptsValidity = np.append(self.ptsValidity, True)
                    cv2.circle(self.geDirMap, (center.y , center.x), 5, (0, 255, 0), 1, 8, 0)
        # Add the LE in input to the current GE .
        if addLeDecision :
            # Save center of mass .
            self.pts = np.append(self.pts, center)
            # Add the LE to the current GE .
            self.LEList = np.append(self.LEList, le)
            # Reset age without any new LE .
            self.geAgeLastLE = 0
            # Update GE map .
            self.geMap = self.geMap + le.getMap()
            # Update colored ge map .
            roiH, roiW = 10, 10
            for pt in le.mLeRoiList :
                roi = np.zeros((roiH, roiW, 3), dtype=np.uint8)
                roi[:,:] = self.geColor
                self.geMapColor[pt.x-roiH//2:pt.x+roiH//2, pt.y-roiW//2:pt.y+roiW//2] = roi
            # Update dirMap .
            self.geDirMap[center.x, center.y] = [0,255,0] 
        else :
            self.geDirMap[center.x, center.y] = [0,0,255]
        
        return True
    
    def ratioFramesDist(self,msg:str) :
        msg += " ratioFrameDist\n"
        dist = sqrt(pow(self.mainPts[-1].x-self.mainPts[0].x,2)+pow(self.mainPts[-1].y-self.mainPts[0].y, 2))
        msg += "d = " + str(dist) + " \n"
        n = self.geLastFrameNum - self.geFirstFrameNum
        msg += "n = " + str(n) +" \n"
        if dist > (n*0.333):
            msg += " ratio = ok \n"
            return (True, msg)
        else:
            msg += " ratio = not ok \n"
            return (False, msg)

    def continuousGoodPos(self, n:int, msg:str):
        msg += "continuousGoodPos\n"
        nb, nn = 0, 0
        msg += "size pts validity : " + str(self.ptsValidity.size) + "\n"
        for i in range(0, self.ptsValidity.size):
            if self.ptsValidity[i] :
                nb += 1
                nn = 0
                if nb >= n :
                    msg += "continuousGoodPos "+ str(n) + " = OK\n"
                    return (True, msg)
            else:
                nn += 1
                nb = 0
                if nn == 2 :
                    msg += "continuousGoodPos "+ str(n) + " =NOT OK\n" 
                    return (False, msg)
        return (False, msg)

    def continuousBadPos(self, n:int):
        nb, nn = 0, 0
        for i in range(0, self.ptsValidity.size):
            if not self.ptsValidity[i] :
                nb += 1
                nn = 0
                if nb >= n :
                    return True
            else:
                nn += 1
                nb = 0
                if nn == 2 :
                    return False
        return False
            

    def negPosClusterFilter(self, msg:str):
        msg += "negPosClusterFilter\n"
        counter = 0
        msg += "clusterNegPos size = "+ str(self.clusterNegPos.size)+"\n"
        for i in range(0, self.clusterNegPos.size):
            if self.clusterNegPos[i]:
                msg += "clusterNegPos true\n"
                counter += 1
            else :
                msg += "clusterNegPos false\n"
        if counter >= float(self.clusterNegPos.size)/2.0 and counter != 0 :
            msg += "negPosClusterFilter = OK"
            return (True, msg)
        else:
            msg +="negPosClusterFilter = NOT OK\n"
            return (False, msg)

    def getMapEvent(self) :
        return self.geMap
    
    def getDirMap(self) :
        return self.geDirMap
    
    def getAge(self) :
        return self.geAge
    
    def getAgeLastElem(self) :
        return self.geAgeLastLE
    
    def getDate(self) :
        return self.geDate
    
    def getLinearStatus(self) :
        return self.geLinear
    
    def getVelocity(self) :
        return self.geShifting
    
    def getNewLEStatus(self) :
        return self.newLeAdded
    
    def getBadPos(self) :
        return self.geBadPoint
    
    def getGoodPos(self) : 
        return self.geGoodPoint
    
    def getNumFirstFrame(self) : 
        return self.geFirstFrameNum
    
    def getNumLastFrame(self) :
        return self.geLastFrameNum
    
    def getGeMapColor(self) :
        return self.geMapColor
    
    def setAge(self, age:int) : 
        self.geAge = age

    def setAgeLastElem(self,age:int) :
        self.geAgeLastLE = age

    def setMapEvent(self, m) :
        self.geMap = m
        
    def setNewLEStatus(self, s:bool) :
        self.newLeAdded = s

    def setNumFirstFrame(self,n:int) :
        self.geFirstFrameNum = n

    def setNumLastFrame(self, n:int) :
        self.geLastFrameNum = n