import argparse
import sys
import os
import shutil
from functools import partial
from datetime import datetime
import time

import cv2

from source.plate_detector import PlateDetector
from source.plate_tracker import PlateTracker
from source.plate_recognizer import PlateRecognizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(input_path, output_path, save, gpu, show):
    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    video_cap = cv2.VideoCapture(input_path)

    plate_detector = PlateDetector()
    plate_tracker = PlateTracker()
    plate_reccognizer = PlateRecognizer()

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        # Detection
        plates = plate_detector.detect(frame)

        for plate in plates:
            for j in range(4):
                p1 = (int(plate.roi[j][0]), int(plate.roi[j][1]))
                p2 = (int(plate.roi[(j + 1) % 4][0]), int(plate.roi[(j + 1) % 4][1]))
                cv2.line(frame, p1, p2, (0, 255, 0), 1)

        # Tracking
        plate_tracker.track(frame, plates)

        for id, item in plate_tracker.history.items():
            x1, y1, x2, y2 = item['plate'].rect
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
            cv2.putText(frame, f"Car_{id}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (127, 255, 0), thickness=2)

            # Recognition
            if item['plate'].rect[1] > frame.shape[0] * 0.5:
                if not item['recognized']:
                    plate_reccognizer.recognize(item['plate'])
                    print(item['plate'].text)
                    if item['plate'].text == None:
                        name = os.path.join(output_path, str(id) + '.jpg')
                    else:
                        item['recognized'] = True
                        name = os.path.join(output_path, item['plate'].text + '.jpg')
                    
                    if save:
                        cv2.imwrite(name, item['plate'].image)
        if show:
            cv2.imshow('output', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Car and License Plate Recognition')
    parser.add_argument('-i', '--input', help="input image or video path", default="IO/input/C0003.mp4", type=str)
    parser.add_argument('-o', '--output', help="output image or video path", default="IO/output", type=str)
    parser.add_argument("-s", "--save", help="whether to save", default=True, action="store_true")
    parser.add_argument('--gpu', action="store_true", default=False, help='Use gpu inference')
    parser.add_argument("--show", help="show live result", default=True, action="store_true")
    args = parser.parse_args()

    main(args.input, args.output, args.save, args.gpu, args.show)
