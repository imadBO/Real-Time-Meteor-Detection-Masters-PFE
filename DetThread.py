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
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture('meteor1.avi')
    
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")
    # Read until video is completed
    while(cap.isOpened()):
        
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        # Display the resulting frame
            frameDate = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            cframe = Frame(frame, 0, 0, frameDate)
            dataQueue.put(cframe)
            cv2.imshow('Frame', frame) 
        # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
    # Break the loop
        else:
            break
    
    # When everything done, release
    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

def startDetection(dataQueue):
    det = TemporalDetection(dtp= detectionParam(), fmt= CamPixFmt.MONO8)
    if dataQueue.empty() :
        time.sleep(1)
    while not dataQueue.empty() :
        det.runDetection(cframe=dataQueue.get())

if __name__ == "__main__" :
    dataQueue = Queue()
    videoStreamProcess = Process(target=playVideo, args=(dataQueue,))
    detectionProcess = Process(target=startDetection, args=(dataQueue,))
    videoStreamProcess.start()
    detectionProcess.start()

    videoStreamProcess.join()
    detectionProcess.join()

    print("detection ended")

