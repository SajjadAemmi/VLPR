import cv2
import numpy as np

outNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
net = cv2.dnn.readNet('../models/plate_detector.pb')
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

image = cv2.imread('../input/2_crop.jpg')

resize_w = 1280
resize_h = 960


# Create a 4D blob from frame.
blob = cv2.dnn.blobFromImage(image, 1.0, (resize_w, resize_h), (123.68, 116.78, 103.94), True, False)

net.setInput(blob)  # Run the detection model

# tickmeter.start()
outs = net.forward(outNames)
# tickmeter.stop()

scores = np.squeeze(outs[0].transpose(0, 2, 3, 1), axis=0)
geometries = np.squeeze(outs[1].transpose(0, 2, 3, 1), axis=0)

print(geometries[0, 0])
print(scores[0, 0])

print(geometries.shape)

image = cv2.resize(image, (resize_w, resize_h))
image_org = image.copy()

count = 0
for i in range(geometries.shape[0]):
    for j in range(geometries.shape[1]):
        if scores[i, j] > 0.99999:
            # print(geometries[i, j, :], 'geo')
            count += 1
            origin = [j*4, i*4]
            d = geometries[i, j, :4]
            angle = geometries[i, j, 4]

            p = np.array([[0, -d[0] - d[2]],
                        [d[1] + d[3], -d[0] - d[2]],
                        [d[1] + d[3], 0],
                        [0, 0],
                        [d[3], -d[2]]])

            rotate_matrix_x = np.repeat([[np.cos(angle)], [np.sin(angle)]], 5, axis=1).transpose((1,0))
            rotate_matrix_y = np.repeat([[-np.sin(angle)], [np.cos(angle)]], 5, axis=1).transpose((1,0))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=1)
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=1)
            p_rotate = np.array([p_rotate_x, p_rotate_y]).transpose((1, 0))
            # print(p_rotate)
            p3_in_origin = origin - p_rotate[4, :]
            new_p = np.array(p_rotate[:4, :] + p3_in_origin, dtype=int)

            # image_org = image.copy()
            print(i, j, new_p)
            for i in range(4):
                cv2.line(image_org, (new_p[i, 0], new_p[i, 1]), (new_p[(i+1) % 4, 0], new_p[(i+1) % 4, 1]), (0, 255, 0), thickness=2)

# cv2.imwrite('../output/1.jpg', image_org)
            cv2.imshow('', image_org)
            cv2.waitKey()
