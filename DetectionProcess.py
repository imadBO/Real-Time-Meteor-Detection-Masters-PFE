from multiprocessing import Process
import TemporalDetection
from TemporalDetection import *
from SaveDetectedEventProcess import *
class DetectionProcess(Process):
    def __init__(self, queue, detParam, fmt, stopEvent):
        Process.__init__(self)
        self.queue = queue
        self.detectionParam = detParam
        self.camPixFmt = fmt
        self.stopEvent = stopEvent

    def run(self):
        det = TemporalDetection(dtp=self.detectionParam, fmt=self.camPixFmt)
        saveQueue = []
        lastTime = time.perf_counter()
        while not self.stopEvent.is_set():
            if self.queue.empty():
                time.sleep(0.1)
                if time.perf_counter() - lastTime > 5 :
                    break
                else :
                    continue
            lastTime = time.perf_counter()
            cframe = self.queue.get(block=False)
            saveQueue.append(cframe)
            _, saving, geToSave = det.runDetection(cframe=cframe)
            Frame.mFrameNumber += 1
            if saving :
                det.resetDetection(False)
                saveEventProcess = SaveDetectedEvent(saveQueue,cframe.mHeight, cframe.mWidth, geToSave)
                saveEventProcess.start()
                # saveEventProcess.join()            
            if len(saveQueue)== 1800:
                saveQueue = saveQueue[900:-1]