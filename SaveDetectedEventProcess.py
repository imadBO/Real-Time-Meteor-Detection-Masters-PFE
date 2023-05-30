from multiprocessing import Process
import cv2 
class SaveDetectedEvent(Process):
    def __init__(self,queue,h,w, geToSave) :
        Process.__init__(self)
        self.queue = queue
        self.h = h
        self.w = w
        self.geToSave = geToSave
    
    def run(self):
        firstFrame = self.geToSave.getNumFirstFrame()
        lastFrame = self.geToSave.getNumLastFrame()
        fi,bi = 0,0
        for i in range(len(self.queue)) :
            if self.queue[i].frameNumber == firstFrame:
                fi = i - 60
            if self.queue[i].frameNumber == lastFrame:
                bi = i + 60
        if fi < 0 :
            fi = 0
        if bi > len(self.queue):
            bi = -1

        saveData = self.queue[fi:bi]
        date = saveData[0].mDate
        fileName = f"event{date.year}-{date.month}-{date.day}-{date.hours}-{date.minutes}-{date.seconds}"
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        writer = cv2.VideoWriter(f"{fileName}.mp4", fourcc, 25.0, (self.h,self.w))
        for elt in saveData :
            cv2.putText(elt.mImg, f"{elt.mDate.year}/{elt.mDate.month}/{elt.mDate.day}  {elt.mDate.hours}:{elt.mDate.minutes}:{elt.mDate.seconds}", (15,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv2.LINE_AA)
            writer.write(elt.mImg)
        writer.release()
