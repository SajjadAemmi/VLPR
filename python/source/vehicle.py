import cv2
import numpy as np


class Vehicle:
    outputSize = (320, 64)

    def __init__(self, rect, class_id=-1, confidence=0):
        self.rect = rect
        self.confidence = confidence
        self.class_id = class_id
        self.not_found = 0
        self.id = id

    def updateImage(self, frame):
        self.image = self.perspectiveTransform(frame, self.roi)
