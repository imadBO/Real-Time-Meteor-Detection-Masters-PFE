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

def playVideo(dataQueue):
    cap = cv2.VideoCapture('PFE/meteor1.avi')

    if not cap.isOpened():
        print("Error opening video file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frameDate = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            cframe = Frame(frame, 0, 0, frameDate)
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
    lastTime = time.perf_counter()
    while True:
        if dataQueue.empty():
            time.sleep(0.1)
            if time.perf_counter() - lastTime > 5 :
                break
            else :
                continue
        lastTime = time.perf_counter()
        # try :
        cframe = dataQueue.get(block=False)
        det.runDetection(cframe=cframe)

if __name__ == "__main__" :
    dataQueue = Queue()
    videoStreamProcess = Process(target=playVideo, args=(dataQueue,))
    detectionProcess = Process(target=startDetection, args=(dataQueue,))
    videoStreamProcess.start()
    detectionProcess.start()

    videoStreamProcess.join()
    detectionProcess.join()

    print("detection ended")

