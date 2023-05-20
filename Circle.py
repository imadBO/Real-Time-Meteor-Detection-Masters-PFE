import cv2
import numpy as np
from math import sqrt, pow, pi, asin, sin
from Types import *

class Circle:
    def __init__(self, center:Point, radius:int):
        self.mPos = center
        self.mRadius = radius
    
    def getCenter(self):
        return self.mPos
    
    def getRadius(self):
        return self.mRadius
    
    def computeDiskSurfaceIntersection(self, circle:"Circle",enableDebug:bool, debugPath:str):
        imgMap = np.zeros((480, 640, 3), dtype=np.uint8)
        res = False
        displayIntersectedSurface = False
        surfaceCircle1 = 0.0
        surfaceCircle2 = 0.0
        intersectedSurface = 0.0

        if enableDebug :
            cv2.circle(imgMap, (int(self.mPos.y), int(self.mPos.x)), int(self.mRadius), (0, 255, 0), thickness=2)
            cv2.circle(imgMap, (int(circle.getCenter().y), int(circle.getCenter().x)), int(circle.getRadius()), (0, 0, 255), thickness=2)

        # Compute the distance between two circle centers .
        circleCentersDist = sqrt(pow(self.mPos.x - circle.getCenter().x, 2) + pow(self.mPos.y - circle.getCenter().y, 2))

        # No intersection .
        if circleCentersDist > circle.getRadius()+self.mRadius :
            if enableDebug:
                cv2.putText(imgMap, "No intersections.", (15,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv2.LINE_AA)
            res = False

        # Circles coincide .
        elif circleCentersDist == 0 and circle.getRadius() == self.mRadius :
            if enableDebug:
                cv2.putText(imgMap, "Circles coincides.", (15, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            res = True

        # A circle is contained inside another .
        elif circleCentersDist < abs(circle.getRadius() - self.mRadius):
            if enableDebug:
                cv2.putText(imgMap, "A circle is contained within the other.", (15, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            res = True
        else :
            surfaceCircle1 = pi * pow(self.mRadius, 2)
            surfaceCircle2 = pi * pow(circle.getRadius(), 2)
            R0 = self.mRadius
            R1 = circle.getRadius()
            x0 = self.mPos.x
            y0 = self.mPos.y
            x1 = circle.getCenter().x
            y1 = circle.getCenter().y

            if self.mPos.y != circle.getCenter().y:
                N = (pow(R1,2) - pow(R0,2) - pow(x1,2) + pow(x0,2) - pow(y1,2) + pow(y0,2)) / (2 * (y0 - y1))
                A = pow((x0-x1)/(y0-y1),2) + 1
                B = 2*y0*((x0-x1)/(y0-y1))-2*N*((x0-x1)/(y0-y1))-2*x0
                C = pow(x0,2) + pow(y0,2) + pow(N,2) - pow(R0,2) - 2* y0*N
                delta = sqrt(pow(B,2)-4*A*C)

                if delta > 0 :
                    resX1 = (-B-delta) / (2*A)
                    resX2 = (-B+delta) / (2*A)
                    resY1 = N - resX1 * ((x0-x1)/(y0-y1))
                    resY2 = N - resX2 * ((x0-x1)/(y0-y1))

                    if enableDebug:
                        cv2.line(imgMap, (int(resX1),int(resY1)), (int(resX2),int(resY2)), (255, 255, 255), 1, cv2.LINE_AA)

                    # Circle1 more inside the other .
                    if circleCentersDist > abs(circle.getRadius() - self.mRadius) and circleCentersDist < circle.getRadius() and circle.getRadius() > self.mRadius:
                        # print("Circle1 more inside the other")
                        # Cord length.
                        c = sqrt(pow((resX1 - resX2),2) + pow((resY1 - resY2),2))
                        cc = c/(2.0*R0)
                        if cc>1.0 : cc = 1.0
                        thetaCircle1 = 2.0* asin(cc)
                        areaCircle1 = (pow(R0,2)/2) * (thetaCircle1 - sin(thetaCircle1))
                        ccc = c/(2.0*R1)
                        if ccc>1.0 : ccc=1.0
                        thetaCircle2 = 2* asin(ccc)
                        areaCircle2 = (pow(R1,2)/2) * (thetaCircle2 - sin(thetaCircle2))
                        intersectedSurface = surfaceCircle1 - areaCircle1 + areaCircle2
                        displayIntersectedSurface = True
                    
                    # Circle2 more inside the other .
                    elif circleCentersDist > abs(circle.getRadius()- self.mRadius) and circleCentersDist < self.mRadius and self.mRadius > circle.getRadius():
                        # print("Circle2 is more inside the other")
                        # Cord length.
                        c = sqrt(pow((resX1 - resX2),2) + pow((resY1 - resY2),2))
                        cc = c/(2.0*R0)
                        if cc>1.0 : cc = 1.0
                        thetaPosCircle = 2.0* asin(cc)
                        areaPosCircle = (pow(R0,2)/2) * (thetaPosCircle - sin(thetaPosCircle))
                        ccc = c/(2.0*R1)
                        if ccc>1.0 : ccc=1.0
                        thetaNegCircle = 2* asin(ccc)
                        areaNegCircle = (pow(R1,2)/2) * (thetaNegCircle - sin(thetaNegCircle))
                        intersectedSurface = surfaceCircle2 - areaNegCircle + areaPosCircle
                        displayIntersectedSurface = True
                    elif circleCentersDist == circle.getRadius() or circleCentersDist == self.mRadius:
                        # print("Outskirt")
                        pass
                    else:
                        c = sqrt(pow((resX1 - resX2),2) + pow((resY1 - resY2),2))
                        cc = c/(2.0*R0)
                        if cc>1.0 : cc = 1.0
                        thetaPosCircle = 2.0* asin(cc)
                        areaPosCircle = (pow(R0,2)/2) * (thetaPosCircle - sin(thetaPosCircle))

                        ccc = c/(2.0*R1)
                        if ccc>1.0 : ccc=1.0
                        thetaNegCircle = 2* asin(ccc)
                        areaNegCircle = (pow(R1,2)/2) * (thetaNegCircle - sin(thetaNegCircle))
                        intersectedSurface = areaNegCircle + areaPosCircle
                        displayIntersectedSurface = True
                    res = True
            else:
                x = (pow(R1,2) - pow(R0,2) - pow(x1,2) + pow(x0,2))/(2*(x0-x1))
                A = 1.0
                B = -2 * y1
                C = pow(x1,2) + pow(x,2) - 2*x1*x + pow(y1,2) - pow(R1,2)
                delta = sqrt(pow(B,2)-4*A*C)

                if delta > 0 :
                    resY1 = (-B-delta) / (2*A)
                    resY2 = (-B+delta) / (2*A)
                    resX1 = (pow(R1,2) - pow(R0,2) - pow(x1,2) + pow(x0,2) - pow(y1,2) + pow(y0,2))/(2*(x0-x1))
                    resX2 = (pow(R1,2) - pow(R0,2) - pow(x1,2) + pow(x0,2) - pow(y1,2) + pow(y0,2))/(2*(x0-x1))

                    if enableDebug:
                        cv2.line(imgMap, (int(resX1), int(resY1)), (int(resX2), int(resY2)), (255, 255, 255), 1, cv2.LINE_AA)

                    # Circle neg more inside the other .
                    if circleCentersDist > abs(circle.getRadius()-self.mRadius and circleCentersDist < circle.getRadius() and circle.getRadius() > self.mRadius):
                        # Cord length .
                        c = sqrt(pow((resX1 - resX2),2) + pow((resY1 - resY2),2))
                        cc = c/(2.0*R0)
                        if cc>1.0 : cc = 1.0
                        thetaPosCircle = 2.0* asin(cc)
                        areaPosCircle = (pow(R0,2)/2) * (thetaPosCircle - sin(thetaPosCircle))
                        ccc = c/(2.0*R1)
                        if ccc>1.0 : ccc=1.0
                        thetaNegCircle = 2* asin(ccc)
                        areaNegCircle = (pow(R1,2)/2) * (thetaNegCircle - sin(thetaNegCircle))
                        intersectedSurface = surfaceCircle1 - areaPosCircle + areaNegCircle
                        displayIntersectedSurface = True
                    
                    # Circle pos more inside the other .
                    elif circleCentersDist > abs(circle.getRadius()- self.mRadius) and circleCentersDist < self.mRadius and self.mRadius < circle.getRadius():
                        # Cord length.
                        c = sqrt(pow((resX1 - resX2),2) + pow((resY1 - resY2),2))
                        cc = c/(2.0*R0)
                        if cc>1.0 : cc = 1.0
                        thetaPosCircle = 2.0* asin(cc)
                        areaPosCircle = (pow(R0,2)/2) * (thetaPosCircle - sin(thetaPosCircle))
                        ccc = c/(2.0*R1)
                        if ccc>1.0 : ccc=1.0
                        thetaNegCircle = 2* asin(ccc)
                        areaNegCircle = (pow(R1,2)/2) * (thetaNegCircle - sin(thetaNegCircle))
                        intersectedSurface = surfaceCircle2 - areaNegCircle + areaPosCircle
                        displayIntersectedSurface = True
                    elif circleCentersDist == circle.getRadius() or circleCentersDist == self.mRadius :
                        # print("The center of one of the circles is on blablabla of the other")
                        pass
                    else :
                        c = sqrt(pow((resX1 - resX2),2) + pow((resY1 - resY2),2));
                        cc = c/(2.0*R0)
                        if cc>1.0 : cc = 1.0
                        thetaPosCircle = 2.0* asin(cc)
                        areaPosCircle = (pow(R0,2)/2) * (thetaPosCircle - sin(thetaPosCircle))
                        ccc = c/(2.0*R1)
                        if ccc>1.0 : ccc=1.0
                        thetaNegCircle = 2* asin(ccc)
                        areaNegCircle = (pow(R1,2)/2) * (thetaNegCircle - sin(thetaNegCircle))
                        intersectedSurface = areaNegCircle + areaPosCircle
                        displayIntersectedSurface = True
                    res = True

        if enableDebug and displayIntersectedSurface:
            cv2.putText(imgMap, "Intersected surface : ", (15,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv2.LINE_AA)
            msg1 = "".join(["- Green circle : " , str((intersectedSurface * 100) / surfaceCircle1) , "%"])
            cv2.putText(imgMap, msg1, (15,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv2.LINE_AA)
            msg2 = "".join(["- Red circle : " , str((intersectedSurface * 100) / surfaceCircle2) , "%"])
            cv2.putText(imgMap, msg2, (15,45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv2.LINE_AA)

        if enableDebug:
            cv2.imwrite(debugPath+'savedimage.jpeg', imgMap)  
            
        return (res, surfaceCircle1, surfaceCircle2, intersectedSurface)