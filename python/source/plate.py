import cv2
import numpy as np


class Plate:
    outputSize = (320, 64)

    def __init__(self, frame, rect, roi=None, confidence=0, id=-1):
        self.rect = rect
        self.roi = roi
        self.confidence = confidence
        self.text = None
        self.not_found = 0
        self.id = id
        self.recognized = False

        if roi is not None:
            self.image = self.perspectiveTransform(frame, self.roi)

    def updateImage(self, frame):
        self.image = self.perspectiveTransform(frame, self.roi)

    def perspectiveTransform(self, frame, roi):
        vertices = np.asarray(roi)
        # outputSize = (100, 32)
        outputSize = (320, 64)
        targetVertices = np.array([[0, 0],
                                   [outputSize[0] - 1, 0],
                                   [outputSize[0] - 1, outputSize[1] - 1],
                                   [0, outputSize[1] - 1]
                                   ],
                                  dtype="float32")

        rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
        result = cv2.warpPerspective(frame, rotationMatrix, outputSize)

        return result
