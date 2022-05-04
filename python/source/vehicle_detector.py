import sys
import os
from functools import partial
from datetime import datetime
import time

import cv2
import numpy as np

from .functions import *
from .vehicle import Vehicle


class VehicleDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet("models/yolov3.cfg", "models/yolov3.weights")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.conf_threshold = 0.7
        self.nms_threshold = 0.45

        self.size = (416, 416)
        self.scalefactor = 1 / 255.0
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(self.scalefactor, self.size)

    def detect(self, image):
        vehicles = []
        class_ids, confidences, boxes = self.model.detect(image, self.conf_threshold, self.nms_threshold)
        for i in range(len(boxes)):
            vehicles.append(Vehicle(rect=boxes[i], class_id=class_ids[i], confidence=confidences[i]))

        return vehicles

