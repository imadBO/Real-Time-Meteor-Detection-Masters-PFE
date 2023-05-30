import wx
from gui import *

if __name__ == "__main__":
    app = wx.App()
    queue = Queue()
    detectionQueue = Queue()
    frame = MyFrame(None,"MeteorDet", queue, detectionQueue)
    frame.Show()
    app.MainLoop()
    frame.panel.video_process.terminate()
    frame.panel.video_process.join()
    


