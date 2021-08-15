import sys
import os
from functools import partial
import time
from sortedcontainers import SortedDict

import cv2
import numpy as np

from .plate import Plate
from .sort import *


class PlateTracker:
    def __init__(self):
        self.not_found_threshold = 10
        self.min_hits = 2
        self.sort = Sort(max_age=self.not_found_threshold, min_hits=self.min_hits)
        self.sort_buffer = SortedDict()
        self.after_counting = SortedDict()
        self.history = {}
        self.top_y_range_ratio = 10
        self.bottom_y_range_ratio = 10

    def track(self, frame, plates):
        detections = []
        for plate in plates:
            center_x = (plate.rect[0] + plate.rect[2]) / 2
            center_y = (plate.rect[1] + plate.rect[3]) / 2

            w = abs(plate.rect[2] - plate.rect[0])
            h = abs(plate.rect[3] - plate.rect[1])

            x1 = center_x - w / 2
            y1 = center_y - h * 2
            x2 = center_x + w / 2
            y2 = center_y + h * 2

            detections.append([int(x1), int(y1), int(x2), int(y2), plate.confidence])

        matches, unmatched_dets, unmatched_trks, track_bbs_ids = self.sort.update(np.array(detections))

        for index, bb in enumerate(track_bbs_ids):  # add all bbox to history
            id = int(bb[-1])
            roi = plates[index].roi
            rect = np.array(bb[:4], dtype='int')

            if frame.shape[0] / self.top_y_range_ratio < bb[1] < frame.shape[0] - frame.shape[0] / self.bottom_y_range_ratio:
                if id not in self.history.keys():  # add new id
                    self.history[id] = {"plate": Plate(frame, rect, roi, id=id), "not_found": 0, "recognized": False}
                    # print(f'plate {id} added to list')
                else:
                    self.history[id]["plate"] = Plate(frame, rect, roi, id=id)
                    self.history[id]["not_found"] = 0

        for i in list(self.history):
            self.history[i]["not_found"] += 1
            if self.history[i]["not_found"] > self.not_found_threshold:
                del self.history[i]
                # print(f'plate {i} remove from list')

        return frame
