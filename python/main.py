import cv2
import numpy as np
from detector_tf import lanms
from detector_tf.icdar import restore_rectangle


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


def resize_image(im, max_side_len=2400):
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


def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.2, nms_thresh=0.2):
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
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    # nms part
    # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the original paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


outNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
net = cv2.dnn.readNet('plate_detector.pb')
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

image = cv2.imread('input/8_crop.jpg')
frame_resized, (ratio_h, ratio_w) = resize_image(image)
height, width = frame_resized.shape[0], frame_resized.shape[1]

print(width, height)
# Create a 4D blob from frame.
blob = cv2.dnn.blobFromImage(image, 1.0, (width, height), (123.68, 116.78, 103.94), True, False)

net.setInput(blob)  # Run the detection model

# tickmeter.start()
outs = net.forward(outNames)
# tickmeter.stop()

scores = outs[0].transpose(0, 2, 3, 1)
geometries = outs[1].transpose(0, 2, 3, 1)

# tickmeter.start()
detected = detect(score_map=scores, geo_map=geometries)

objects = []
if detected is not None:
    confidences = detected[:, 8]
    boxes = detected[:, :8].reshape((-1, 4, 2))
    boxes[:, :, 0] /= ratio_w
    boxes[:, :, 1] /= ratio_h

    for i, box in enumerate(boxes):
        # print(box)

        for i in range(4):
            cv2.line(image, (box[i][0], box[i][1]), (box[(i+1) % 4][0], box[(i+1) % 4][1]), (0, 255, 0), 2)

        plate_image = fourPointsTransform(image, box)  # get cropped image using perspective transform

        # cv2.imshow('out', plate_image)
        # cv2.waitKey()


cv2.imwrite('output/8_crop.jpg', image)
# cv2.imshow('out', image)
# cv2.waitKey()
