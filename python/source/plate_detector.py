import sys
import os
from functools import partial
from datetime import datetime
import time

import cv2
import numpy as np

from . import lanms
from .functions import *
from .plate import Plate


class PlateDetector:
    def __init__(self):
        self.net = cv2.dnn.readNet("models/plate_detector.pb")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.outNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        self.score_map_thresh = 0.9

    def setVideoPath(self, video_path):
        self.video_path = video_path
        self.video_cap = cv2.VideoCapture(self.video_path)

    def detect(self, frame):
        plates = []

        frame_resized, (ratio_h, ratio_w) = self.resize_image(frame)
        height, width = frame_resized.shape[0], frame_resized.shape[1]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height), (123.68, 116.78, 103.94), True, False)
        self.net.setInput(blob)
        outs = self.net.forward(self.outNames)

        scores = outs[0].transpose(0, 2, 3, 1)
        geometries = outs[1].transpose(0, 2, 3, 1)
        detected = self.postProcess(score_map=scores, geo_map=geometries, score_map_thresh=self.score_map_thresh)

        if detected is not None:
            confidences = detected[:, 8]
            rois = detected[:, :8].reshape((-1, 4, 2))
            rois[:, :, 0] /= ratio_w
            rois[:, :, 1] /= ratio_h

            for i, roi in enumerate(rois):
                xyxy = convert_4boxPoints_to_xyxy(roi)
                plates.append(Plate(frame, rect=xyxy, roi=roi, confidence=confidences[i]))

        return plates

    def resize_image(self, im, max_side_len=2400):
        '''
        resize image to a size multiple of 32 which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        '''
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
        im = cv2.resize(im, (int(resize_w), int(resize_h)))

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)

    def postProcess(self, score_map, geo_map, score_map_thresh=0.8, box_thresh=0.2, nms_thresh=0.2):
        '''
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param timer:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thresh: threshold for nms
        :return:
        '''
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]

        # restore
        text_box_restored = self.restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

        # nms part
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)

        if boxes.shape[0] == 0:
            return None

        # here we filter some low score boxes by the average score map, this is different from the original paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes

    def restore_rectangle(self, origin, geometry):
        d = geometry[:, :4]
        angle = geometry[:, 4]
        # for angle > 0
        origin_0 = origin[angle >= 0]
        d_0 = d[angle >= 0]
        angle_0 = angle[angle >= 0]
        if origin_0.shape[0] > 0:
            p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                          d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                          d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                          np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                          d_0[:, 3], -d_0[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_0 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_0 = np.zeros((0, 4, 2))
        # for angle < 0
        origin_1 = origin[angle < 0]
        d_1 = d[angle < 0]
        angle_1 = angle[angle < 0]
        if origin_1.shape[0] > 0:
            p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                          np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                          np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                          -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                          -d_1[:, 1], -d_1[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_1 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_1 = np.zeros((0, 4, 2))
        return np.concatenate([new_p_0, new_p_1])