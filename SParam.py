import numpy as np
from Mask import *
import Mask
from ETimeMode import TimeMode
from EDetMeth import DetMeth
from EStackMeth import StackMeth

##########################################################
#                 Detection parameters                   #
##########################################################
class detectionParam:
    def __init__(self):
        self.ACQ_BUFFER_SIZE:int = 0
        self.ACQ_MASK_ENABLED:bool = False
        self.ACQ_MASK_PATH:str = ""
        self.MASK:Mask = None
        self.DET_ENABLED:bool = False
        self.DET_MODE:TimeMode = None
        self.DET_DEBUG:bool = True # Change it back to false .
        self.DET_DEBUG_PATH:str = "PFE"
        self.DET_TIME_AROUND:int = 0
        self.DET_TIME_MAX:int = 10000
        self.DET_METHOD:DetMeth = None
        self.DET_SAVE_FITS3D:bool = False
        self.DET_SAVE_FITS2D:bool = False
        self.DET_SAVE_SUM:bool = False
        self.DET_SUM_REDUCTION:bool = False
        self.DET_SUM_MTHD:StackMeth = None
        self.DET_SAVE_SUM_WITH_HIST_EQUALIZATION:bool = False
        self.DET_SAVE_AVI:bool = False
        self.DET_UPDATE_MASK:bool = False
        self.DET_UPDATE_MASK_FREQUENCY:int = 10
        self.DET_DEBUG_UPDATE_MASK:bool = False
        self.DET_DOWNSAMPLE_ENABLED:bool = True

        self.temporal = self.DetectionMethod1()
        
        self.status:bool = False
        self.errormsg = np.array([])
    class DetectionMethod1:
        def __init__(self):

            self.DET_SAVE_GEMAP:bool = False
            self.DET_SAVE_DIRMAP:bool = False
            self.DET_SAVE_POS:bool = False
            self.DET_LE_MAX:int = 9 # From 1 to 10 .
            self.DET_GE_MAX:int = 9
            # self.DET_SAVE_GE_INFOS:bool = False


##########################################################
#                 Station parameters                     #
##########################################################
class StationParam:
    def __init__(self):
        self.STATION_NAME = ""
        self.TELESCOP = ""
        self.OBSERVER = ""
        self.INSTRUME = ""
        self.CAMERA = ""
        self.FOCAL = 0.0
        self.APERTURE = 0.0
        self.SITELONG = 0.0
        self.SITELAT = 0.0
        self.SITEELEV = 0.0
        self.status = False
        self.errormsg = []

##########################################################
#                 Fits Keys parameters                   #
##########################################################
class FitskeysParam:
    def __init__(self):
        self.FILTER = ""
        self.K1 = 0.0
        self.K2 = 0.0
        self.COMMENT = ""
        self.CD1_1 = 0.0
        self.CD1_2 = 0.0
        self.CD2_1 = 0.0
        self.CD2_2 = 0.0
        self.XPIXEL = 0.0
        self.YPIXEL = 0.0
        self.status = False
        self.errormsg = []

##########################################################
#                 output data parameters                 #
##########################################################
class DataParam:
    def __init__(self):
        self.DATA_PATH = ""
        self.FITS_COMPRESSION = False
        self.FITS_COMPRESSION_METHOD = ""
        self.status = False
        self.errormsg = []