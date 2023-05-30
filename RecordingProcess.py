from multiprocessing import Process
from datetime import datetime
import time
import cv2

class RecordingProcess(Process):
    def __init__(self, queue, h, w):
        super().__init__()
        self.queue = queue
        self.h = h
        self.w = w

    def run(self):
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(f"PFE/Recording-{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}.mp4", fourcc, 25.0, (self.h,self.w))
        while True :
            if self.queue.empty():
                time.sleep(0.1)
                continue
            cframe = self.queue.get()
            cv2.putText(cframe.mImg, f"{cframe.mDate.year}/{cframe.mDate.month}/{cframe.mDate.day}  {cframe.mDate.hours}:{cframe.mDate.minutes}:{cframe.mDate.seconds}", (15,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv2.LINE_AA)
            writer.write(cframe.mImg)