from ECamPixFmt import CamPixFmt
from TimeDate import TimeDate

class Frame:
    mFrameNumber = 0 # Each frame is identified by a number corresponding to the acquisition order
    def __init__(self, capImg, g, e, acquisitionDate):
        self.mDate:TimeDate = TimeDate.splitIsoExtendedDate(acquisitionDate) # Acquisition date
        # self.mDate = acquisitionDate
        self.mExposure = e # Camera's exposure value used to grab the frame
        self.mGain = g # Camera's gain value used to grab the frame
        self.mFormat:CamPixFmt = CamPixFmt.MONO8 # Pixel format
        self.mImg = capImg # Frame's image data
        self.mFileName = "" # Frame's name
        self.mFrameNumber = 0 # Each frame is identified by a number corresponding to the acquisition order
        self.mFrameRemaining = 0 # Define the number of remaining frames if the input source is a video or a set of single frames
        self.mSaturatedValue = 255 # Max pixel value in the image
        self.mFps = 0.0 # Camera's fps
        self.mWidth = 0
        self.mHeight = 0

    # def __init__(self):
    #     self.mDate:TimeDate = None # Acquisition date
    #     self.mExposure = 0 # Camera's exposure value used to grab the frame
    #     self.mGain = 0 # Camera's gain value used to grab the frame
    #     self.mFormat:CamPixFmt = CamPixFmt.MONO8 # Pixel format
    #     self.mImg = None # Frame's image data
    #     self.mFileName = "" # Frame's name
    #     self.mFrameNumber = 0 # Each frame is identified by a number corresponding to the acquisition order
    #     self.mFrameRemaining = 0 # Define the number of remaining frames if the input source is a video or a set of single frames
    #     self.mSaturatedValue = 255 # Max pixel value in the image
    #     self.mFps = 0.0 # Camera's fps
    #     self.mWidth = 0
    #     self.mHeight = 0
