from mss import mss
import numpy as np
import cv2
from config import GAME_WINDOW

class ScreenCapture:
    def __init__(self):
        self.sct = mss()

    def capture_frame(self):
        screenshot = self.sct.grab(GAME_WINDOW)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        return frame