# importing libraries
from multiprocessing import Process, Queue
import cv2
import numpy as np
import TemporalDetection
from TemporalDetection import *
from SParam import *
from ECamPixFmt import * 
import Frame
from Frame import *
from datetime import datetime

def saveEvent(dataQueue,h,w, geToSave):
    firstFrame = geToSave.getNumFirstFrame()
    lastFrame = geToSave.getNumLastFrame()
    fi,bi = 0,0
    for i in range(len(dataQueue)) :
        if dataQueue[i].frameNumber == firstFrame:
            fi = i - 60
        if dataQueue[i].frameNumber == lastFrame:
            bi = i + 60
    if fi < 0 :
        fi = 0
    if bi > len(dataQueue):
        bi = -1

    saveData = dataQueue[fi:bi]
    date = saveData[0].mDate
    fileName = f"event{date.year}-{date.month}-{date.day}-{date.hours}-{date.minutes}-{date.seconds}"
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter(f"PFE/{fileName}.mp4", fourcc, 25.0, (h,w))
    for elt in saveData :
        cv2.putText(elt.mImg, f"{elt.mDate.year}/{elt.mDate.month}/{elt.mDate.day}  {elt.mDate.hours}:{elt.mDate.minutes}:{elt.mDate.seconds}", (15,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv2.LINE_AA)
        writer.write(elt.mImg)
    writer.release()

def playVideo(dataQueue):
    cap = cv2.VideoCapture('PFE/4.m4v')

    if not cap.isOpened():
        print("Error opening video file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frameDate = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            cframe = Frame(frame, 0, 0, frameDate)
            cframe.mHeight = int(cap.get(3))
            cframe.mWidth = int(cap.get(4))
            cframe.frameNumber = Frame.mFrameNumber
            dataQueue.put(cframe)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def startDetection(dataQueue):
    det = TemporalDetection(dtp=detectionParam(), fmt=CamPixFmt.MONO8)
    saveQueue = []
    lastTime = time.perf_counter()
    while True:
        if dataQueue.empty():
            time.sleep(0.1)
            if time.perf_counter() - lastTime > 5 :
                break
            else :
                continue
        lastTime = time.perf_counter()
        cframe = dataQueue.get(block=False)
        saveQueue.append(cframe)
        _, saving, geToSave = det.runDetection(cframe=cframe)
        Frame.mFrameNumber += 1
        if saving :
            det.resetDetection(False)
            saveEventProcess = Process(target=saveEvent, args=(saveQueue,cframe.mHeight, cframe.mWidth, geToSave))
            saveEventProcess.start()
            saveEventProcess.join()            
        if len(saveQueue)== 1800:
            saveQueue = saveQueue[900:-1]

if __name__ == "__main__" :
    dataQueue = Queue()
    videoStreamProcess = Process(target=playVideo, args=(dataQueue,))
    detectionProcess = Process(target=startDetection, args=(dataQueue,))
    videoStreamProcess.start()
    detectionProcess.start()

    videoStreamProcess.join()
    detectionProcess.join()

    print("detection ended")

