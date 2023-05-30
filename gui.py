import wx
import cv2
from multiprocessing import Queue, Event
from SParam import *
from ECamPixFmt import * 
from datetime import datetime 
from DetectionProcess import *
from VideoCaptureProcess import *
from Cfg import *


class MyFrame(wx.Frame):
    def __init__(self, parent, title, queue, detectionQueue):
        super(MyFrame, self).__init__(parent, title= title, size= (600,500), style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        self.panel = MyPanel(self,queue,detectionQueue)

class MyPanel(wx.Panel):
    def __init__(self, parent, frame_queue,detectionQueue):
        super(MyPanel, self).__init__(parent)
        # Create a reference to detection process to be used later .
        self.detParam = ConfigFiles.load_object("detCfg.pkl")
        self.cameraParam = ConfigFiles.load_object("cameraCfg.pkl")
        self.camPixFmt = CamPixFmt.MONO8
        self.stopDetEvent = Event()
        self.detectionProcess = None
        # Create a timer to update the frame display
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_frame, self.timer)
        self.timer.Start(1000 // int(self.cameraParam.FPS))  # 30 FPS
        # Set up the frame queue .
        self.frame_queue = frame_queue
        self.detectionQueue = detectionQueue
        self.addFrameForDetection = False
        self.continuousCapture = False
        self.fourcc = None
        self.writer = None
        self.initWriter = False
        # Create a stop event to signal the video capture process to stop
        self.stop_event = Event()
        # Start the video capture process
        self.video_process = VideoCaptureProcess(self.frame_queue, self.stop_event)
        self.video_process.start()
        # Bind the close event
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Sizers .
        self.configSizer = wx.BoxSizer(wx.VERTICAL)
        self.gainSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.fpsSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.exposureSizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.buttonsSizer = wx.GridSizer(2,2,15,15)
        self.videoPanelSizer = wx.BoxSizer(wx.VERTICAL)
        self.globalSizer = wx.BoxSizer(wx.HORIZONTAL)

        # Labels .
        self.gainLabel = wx.StaticText(self, label="Gain :")
        self.fpsLabel = wx.StaticText(self, label="Fps :")
        self.exposureLabel = wx.StaticText(self, label="Exposure :")

        # Text inputs .
        self.gainTxtCtrl = wx.TextCtrl(self,)
        self.gainTxtCtrl.SetValue(self.cameraParam.GAIN)
        self.gainTxtCtrl.Disable()
        self.fpsTxtCtrl = wx.TextCtrl(self,)
        self.fpsTxtCtrl.SetValue(self.cameraParam.FPS)
        self.fpsTxtCtrl.Disable()
        self.exposureTxtCtrl = wx.TextCtrl(self,)
        self.exposureTxtCtrl.SetValue(self.cameraParam.EXPOSURE)
        self.exposureTxtCtrl.Disable()
        # Text inputs checkboxes .
        self.autoGainCb = wx.CheckBox(self,label="Auto")
        self.autoGainCb.SetValue(True)
        self.autoGainCb.Bind(wx.EVT_CHECKBOX, self.onAutoGain)
        self.autoFpsCb = wx.CheckBox(self, label="Auto")
        self.autoFpsCb.SetValue(True)
        self.autoFpsCb.Bind(wx.EVT_CHECKBOX, self.onAutoFps)
        self.autoExposureCb = wx.CheckBox(self, label= "Auto")
        self.autoExposureCb.SetValue(True)
        self.autoExposureCb.Bind(wx.EVT_CHECKBOX, self.onAutoExposure)

        # Config parameters .
        self.detDebugCb = wx.CheckBox(self,label="Detection Debug")
        self.detDebugCb.SetValue(self.detParam.DET_DEBUG)
        self.maskCb = wx.CheckBox(self, label="Apply mask")
        self.maskCb.SetValue(self.detParam.DET_UPDATE_MASK)
        self.downsampleCb = wx.CheckBox(self, label= "Enable downsampling")
        self.downsampleCb.SetValue(self.detParam.DET_DOWNSAMPLE_ENABLED)

        # Buttons .
        self.configButton = wx.Button(self, label="Save config")
        self.configButton.Bind(wx.EVT_BUTTON, self.onSaveConfig)
        self.startDetButton = wx.Button(self, label="Start detection")
        self.startDetButton.Bind(wx.EVT_BUTTON,self.onStartDetection)
        self.stopDetButton = wx.Button(self, label="Stop detection")
        self.stopDetButton.Disable()
        self.stopDetButton.Bind(wx.EVT_BUTTON,self.onStopDetection)
        self.startRecordingButton = wx.Button(self, label= "Start recording")
        self.startRecordingButton.Bind(wx.EVT_BUTTON,self.onStartRecording)
        self.stopRecordingButton = wx.Button(self, label= "Stop recording")
        self.stopRecordingButton.Bind(wx.EVT_BUTTON,self.onStopRecording)
        self.stopRecordingButton.Disable()

        # Video panel .
        self.videoPlayer = wx.StaticBitmap(self, size=(350,330))

        # Putting it all together .
        # Config panel .
        self.gainSizer.Add(self.gainTxtCtrl,0, wx.ALL, border=2)
        self.gainSizer.Add(self.autoGainCb, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, border = 2)
        self.configSizer.Add(self.gainLabel, 0, wx.ALL, border = 5)
        self.configSizer.Add(self.gainSizer, 0, wx.ALL)

        self.fpsSizer.Add(self.fpsTxtCtrl, 0, wx.ALL, border = 2)
        self.fpsSizer.Add(self.autoFpsCb, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, border = 2)
        self.configSizer.Add(self.fpsLabel, 0, wx.ALL, border = 5)
        self.configSizer.Add(self.fpsSizer, 0, wx.ALL)

        self.exposureSizer.Add(self.exposureTxtCtrl, 0, wx.ALL, border = 2)
        self.exposureSizer.Add(self.autoExposureCb, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, border = 2)
        self.configSizer.Add(self.exposureLabel, 0, wx.ALL, border = 5)
        self.configSizer.Add(self.exposureSizer, 0, wx.ALL)

        self.configSizer.Add(self.detDebugCb, 0, wx.ALL, border = 5)
        self.configSizer.Add(self.maskCb, 0, wx.ALL, border = 5)
        self.configSizer.Add(self.downsampleCb, 0, wx.ALL, border = 5)
        self.configSizer.Add(self.configButton, 0, wx.ALL, border = 5)

        # Video buttons .
        self.buttonsSizer.Add(self.startDetButton, 0, wx.ALL, border = 5)
        self.buttonsSizer.Add(self.stopDetButton, 0, wx.ALL, border = 5)
        self.buttonsSizer.Add(self.startRecordingButton, 0, wx.ALL, border = 5)
        self.buttonsSizer.Add(self.stopRecordingButton, 0, wx.ALL, border = 5)

        # The whole video panel .
        self.videoPanelSizer.Add(self.buttonsSizer, 0, wx.ALL, border = 5)
        self.videoPanelSizer.Add(self.videoPlayer, 0, wx.ALL, border = 5)

        # The whole window .
        self.globalSizer.Add(self.configSizer, 0, wx.ALL, border = 5)
        self.globalSizer.Add(self.videoPanelSizer, 0, wx.ALL, border = 5)

        self.SetSizer(self.globalSizer)
        self.Fit()

    def update_frame(self, event):
        # Get the latest frame from the queue
        if not self.frame_queue.empty():
            cframe = self.frame_queue.get()
            if self.addFrameForDetection :
                self.detectionQueue.put(cframe)
            elif self.continuousCapture :
                if self.initWriter :
                    self.fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
                    self.writer = cv2.VideoWriter(f"Recording-{datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}.mp4", self.fourcc, 25.0, (cframe.mImg.shape[0],cframe.mImg.shape[1]))
                    self.initWriter = False
                cv2.putText(cframe.mImg, f"{cframe.mDate.year}/{cframe.mDate.month}/{cframe.mDate.day}  {cframe.mDate.hours}:{cframe.mDate.minutes}:{cframe.mDate.seconds}", (15,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1, cv2.LINE_AA)
                self.writer.write(cframe.mImg)
            elif self.writer is not None :
                self.writer.release()
                self.writer = None

            frame = cframe.mImg
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            # Create a wx.Image from the frame
            image = wx.Image(width, height, image)
            # Get the size of the widget
            widget_size = self.videoPlayer.GetSize()
            # Scale the image to fit the widget size
            image = image.Scale(widget_size.width, widget_size.height, wx.IMAGE_QUALITY_HIGH)
            bitmap = wx.Bitmap(image)
            self.videoPlayer.SetBitmap(bitmap)

    def on_close(self, event):
        # Stop the timer, set the stop event, and close the application
        self.timer.Stop()
        self.stop_event.set()
        self.stopDetEvent.set()
        self.detectionProcess.join()
        self.Destroy()

    def onSaveConfig(self, event):
        validInputs = self.validateInputs()
        if validInputs :
            print("Config Saved")
            self.cameraParam.GAIN = self.gainTxtCtrl.GetValue()
            self.cameraParam.FPS = self.fpsTxtCtrl.GetValue()
            self.cameraParam.EXPOSURE = self.exposureTxtCtrl.GetValue()
            self.detParam.DET_DEBUG = self.detDebugCb.GetValue()
            self.detParam.DET_UPDATE_MASK = self.maskCb.GetValue()
            self.detParam.DET_DOWNSAMPLE_ENABLED = self.downsampleCb.GetValue()
            ConfigFiles.save_object(self.detParam,"detCfg.pkl")
            ConfigFiles.save_object(self.cameraParam, "cameraCfg.pkl")
        else :
            wx.MessageBox("'GAIN','FPS','EXPOSURE' should be integers.", "Invalid Input!", wx.OK | wx.ICON_ERROR)
    
    def validateInputs(self):
        try:
            # Try converting values to int.
            int(self.gainTxtCtrl.GetValue())  
            int(self.fpsTxtCtrl.GetValue())
            int(self.exposureTxtCtrl.GetValue())
        except ValueError:
            return False  # Invalid input
        return True  # Valid input

    def onStartDetection(self, event):
        self.addFrameForDetection =  True
        self.detectionProcess = DetectionProcess(self.detectionQueue,self.detParam,self.camPixFmt,self.stopDetEvent)
        self.startDetButton.Disable()
        self.stopDetButton.Enable()
        self.startRecordingButton.Disable()
        self.stopRecordingButton.Disable()
        self.detectionProcess.start()

    def onStopDetection(self,event):
        self.addFrameForDetection = False
        self.startDetButton.Enable()
        self.stopDetButton.Disable()
        self.startRecordingButton.Enable()
        self.stopRecordingButton.Disable()
        # Stop the detection process
        self.stopDetEvent.set()

    def onStartRecording(self,event):
        self.initWriter = True
        self.continuousCapture = True
        self.startDetButton.Disable()
        self.stopDetButton.Disable()
        self.startRecordingButton.Disable()
        self.stopRecordingButton.Enable()
        
    def onStopRecording(self,event):
        self.continuousCapture = False
        self.startDetButton.Enable()
        self.stopDetButton.Disable()
        self.startRecordingButton.Enable()
        self.stopRecordingButton.Disable()

    def onAutoGain(self,event):
        if self.autoGainCb.IsChecked() :
            self.gainTxtCtrl.Disable()
        else :
            self.gainTxtCtrl.Enable()

    def onAutoFps(self,event):
        if self.autoFpsCb.IsChecked() :
            self.fpsTxtCtrl.Disable()
        else :
            self.fpsTxtCtrl.Enable()

    def onAutoExposure(self,event):
        if self.autoExposureCb.IsChecked() :
            self.exposureTxtCtrl.Disable()
        else :
            self.exposureTxtCtrl.Enable()
