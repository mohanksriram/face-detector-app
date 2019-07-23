import cv2
from enum import Enum

"""
Setup a camera source with address to easily access frames
"""

class ECameraMode(Enum):
    SINGLE_FRAME = 0
    CONTINUOUS = 1 

class CameraSource(object):

    def __init__(self, address):
        self.address = address
        self.is_running = False
        self.mode = ECameraMode.SINGLE_FRAME
        self.cam = cv2.VideoCapture(self.address)

    def getStatus(self):
        return self.is_running

    
    def getFrame(self):
        self.is_runnnig = True
        success, img = self.cam.read()
        self.cam_h, self.cam_w, _ = img.shape
        self.is_running = False

        if success:
            return img
        else:
            print("Couldn't fetch frame")
            return None

    def getInfo(self):
        return {'camera': {'width': self.cam_w, 'height': self.cam_h, 'address': self.address}}

