from multiprocessing import Process
from datetime import datetime
import cv2
from Frame import *
class VideoCaptureProcess(Process):
    def __init__(self, frame_queue, stop_event):
        Process.__init__(self)
        self.queue = frame_queue
        self.stop_event = stop_event

    def run(self):
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
                self.queue.put(cframe)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()