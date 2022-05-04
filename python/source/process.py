import sys
import os

from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QProgressBar, QPushButton, QWidget
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QThread, Signal, Qt, QLine
from PySide6.QtGui import QIcon
from PySide6 import QtGui
from functools import partial
from datetime import datetime
import time
import cv2

from .plate_detector import PlateDetector
from .plate_tracker import PlateTracker
from .plate_recognizer import PlateRecognizer
from .vehicle_detector import VehicleDetector


class Process(QThread):
    signalShowPreview = Signal(object)
    signalShowPlate = Signal(object)
    signalUpdateClassCounters = Signal(int)

    def __init__(self):
        QThread.__init__(self)
        self.video_cap = None
        self.input_file_path = None
        self.plate_detector = PlateDetector()
        self.plate_recognizer = PlateRecognizer()
        self.vehicle_detector = VehicleDetector()

    def setInputFilePath(self, input_file_path):
        self.input_file_path = input_file_path

    def processImage(self, image):
        image_original = image

        # Plate Detection
        plates = self.plate_detector.detect(image_original)
        for id, plate in enumerate(plates):
            for j in range(4):
                p1 = (int(plate.roi[j][0]), int(plate.roi[j][1]))
                p2 = (int(plate.roi[(j + 1) % 4][0]), int(plate.roi[(j + 1) % 4][1]))
                cv2.line(image, p1, p2, (255, 0, 0), 4)

            # Plate Recognition
            self.plate_recognizer.recognize(plate)
            if plate.text is not None:
                plate.recognized = True
                self.signalShowPlate.emit(plate)

        # Vehicle Detection
        vehicles = self.vehicle_detector.detect(image_original)
        for id, vehicle in enumerate(vehicles):
            x, y, w, h = vehicle.rect
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 4)
            self.signalUpdateClassCounters.emit(vehicle.class_id)

        self.signalShowPreview.emit(image)

    def run(self):
        print(self.input_file_path)
        file_name, file_extension = os.path.splitext(self.input_file_path)
        if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            image = cv2.imread(self.input_file_path)
            self.processImage(image)
        elif file_extension.lower() in ['.mp4', '.mov', '.mkv']:
            self.video_cap = cv2.VideoCapture(self.input_file_path)
            while True:
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                self.signalShowPreview.emit(frame)
        else:
            print('error! input file extension is not supported')
