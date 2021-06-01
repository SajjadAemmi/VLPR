import numpy as np
import cv2


def besco(origin, angle, p):
    if origin.shape[0] > 0:
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(np.abs(angle)), np.sin(np.abs(angle))]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(np.abs(angle)), np.cos(np.abs(angle))]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p = np.zeros((0, 4, 2))
    
    return new_p


def restore_rectangle(origin, geometry):
    d = geometry[:, :4]
    print(geometry.shape)
    angle = geometry[:, 4]

    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]

    p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2], d_0[:, 1] + d_0[:, 3], 
                    -d_0[:, 0] - d_0[:, 2], d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]), 
                    np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]), d_0[:, 3], -d_0[:, 2]])
                    
    new_p_0 = besco(origin_0, angle_0, p)

    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]

    p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2], np.zeros(d_1.shape[0]), 
                    -d_1[:, 0] - d_1[:, 2], np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]), 
                    -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]), -d_1[:, 1], -d_1[:, 2]])

    new_p_1 = besco(origin_1, angle_1, p)

    return np.concatenate([new_p_0, new_p_1])


def detect(score_map, geo_map, score_map_thresh=0.95, box_thresh=0.1, nms_thres=0.2):
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2

    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    # for i, box in enumerate(boxes):
    #     mask = np.zeros_like(score_map, dtype=np.uint8)
    #     cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
    #     boxes[i, 8] = cv2.mean(score_map, mask)[0]

    # boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


net = cv2.dnn.readNet("model.pb")
outNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

image = cv2.imread("input/3.jpg")

input_width = 1024
input_height = 768

# Create a 4D blob from frame.
blob = cv2.dnn.blobFromImage(image, 1.0, (input_width, input_height), (123.68, 116.78, 103.94), True, False)
net.setInput(blob)  # Run the detection model
outs = net.forward(outNames)

scores = np.squeeze(outs[0].transpose(0, 2, 3, 1))
geometries = np.squeeze(outs[1].transpose(0, 2, 3, 1), axis=0)

detected = detect(score_map=scores, geo_map=geometries)

image = cv2.resize(image, (input_width, input_height))

if detected is not None:
    boxes = detected[:, :8].reshape((-1, 4, 2))

    for box in boxes:
        for i in range(4):
            p1 = (box[i][0], box[i][1])
            p2 = (box[(i + 1) % 4][0], box[(i + 1) % 4][1])
            cv2.line(image, p1, p2, (0, 255, 0), 1)

cv2.imshow("result", image)
cv2.waitKey()
