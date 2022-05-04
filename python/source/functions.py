import numpy as np
import cv2


def alphabetDict(c):
    dict = {
        'A': 'الف',
        'P': 'پ',
        'T': 'ت',
        'Y': 'ث',
        'Z': 'ز',
        'X': 'ژ',
        '#': 'ش',
        'E': 'ع',
        'F': 'ف',
        'K': 'ک',
        'G': 'گ',
        'D': 'D',
        'S': 'S',
        'B': 'ب',
        'J': 'ج',
        'W': 'د',
        'C': 'س',
        'U': 'ص',
        'R': 'ط',
        'Q': 'ق',
        'L': 'ل',
        'M': 'م',
        'N': 'ن',
        'V': 'و',
        'H': 'ه',
        'I': 'ی',
        'O': 'O',
    }
    return dict[c]


def convert_4boxPoints_to_xywh(vertices):
    x1 = np.min(vertices[:, 0])
    y1 = np.min(vertices[:, 1])
    x2 = np.max(vertices[:, 0])
    y2 = np.max(vertices[:, 1])

    w = x2 - x1
    h = y2 - y1

    return [x1, y1, w, h]


def convert_4boxPoints_to_xyxy(vertices):
    x1 = int(np.min(vertices[:, 0]))
    y1 = int(np.min(vertices[:, 1]))
    x2 = int(np.max(vertices[:, 0]))
    y2 = int(np.max(vertices[:, 1]))

    return (x1, y1, x2, y2)


def cal_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    i = max(0, (x2 - x1)) * max(0, (y2 - y1))
    u = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
        (box2[2] - box2[0]) * (box2[3] - box2[1]) - i
    iou = float(i) / float(u)
    return iou


def get_objName(item, objects):
    iou_list = []
    for i, object in enumerate(objects):
        x, y, w, h = object[2]
        x1, y1, x2, y2 = int(x - w / 2), int(y - h /
                                             2), int(x + w / 2), int(y + h / 2)
        iou_list.append(cal_iou(item[:4], [x1, y1, x2, y2]))
    max_index = iou_list.index(max(iou_list))
    return objects[max_index][0]


def filiter_out_repeat(objects):
    objects = sorted(objects, key=lambda x: x[1])
    l = len(objects)
    new_objects = []
    if l > 1:
        for i in range(l - 1):
            flag = 0
            for j in range(i + 1, l):
                x_i, y_i, w_i, h_i = objects[i][2]
                x_j, y_j, w_j, h_j = objects[j][2]
                box1 = [int(x_i - w_i / 2), int(y_i - h_i / 2),
                        int(x_i + w_i / 2), int(y_i + h_i / 2)]
                box2 = [int(x_j - w_j / 2), int(y_j - h_j / 2),
                        int(x_j + w_j / 2), int(y_j + h_j / 2)]
                if cal_iou(box1, box2) >= 0.5:
                    flag = 1
                    break
            # if no repeat
            if not flag:
                new_objects.append(objects[i])
        # add the last one
        new_objects.append(objects[-1])
    else:
        return objects

    return list(tuple(new_objects))


def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices)
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
