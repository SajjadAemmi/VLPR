import argparse
import os
import time
from pathlib import Path

import cv2

from source.plate_detector import PlateDetector
from source.plate_recognizer import PlateRecognizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    output_dir_path = os.path.join(args.output, Path(args.input).stem)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    plate_detector = PlateDetector()
    plate_recognizer = PlateRecognizer()

    image = cv2.imread(args.input)
    image_original = image

    # Plate Detection
    plates = plate_detector.detect(image_original)

    for id, plate in enumerate(plates):
        for j in range(4):
            p1 = (int(plate.roi[j][0]), int(plate.roi[j][1]))
            p2 = (int(plate.roi[(j + 1) % 4][0]), int(plate.roi[(j + 1) % 4][1]))
            cv2.line(image, p1, p2, (255, 0, 0), 4)

        # Plate Recognition
        plate_recognizer.recognize(plate)
        print(plate.text)
        if args.save:
            if plate.text is None:
                name = os.path.join(output_dir_path, f'plate_{id}.jpg')
            else:
                plate.recognized = True
                name = os.path.join(output_dir_path, plate.text + '.jpg')
            cv2.imwrite(name, plate.image)

    if args.save:
        cv2.imwrite(os.path.join(output_dir_path, Path(args.input).stem + '.jpg'), image)

    if args.show:
        cv2.imshow('output', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vehicle and License Plate Recognition')
    parser.add_argument('-i', '--input', help="input image path", default="IO/input/7.bmp", type=str)
    parser.add_argument('-o', '--output', help="output directory path", default="IO/output", type=str)
    parser.add_argument("--save", default=True, action="store_true", help="whether to save")
    parser.add_argument("--gpu", default=False, action="store_true", help='Use gpu inference')
    parser.add_argument("--show", default=False, action="store_true", help="show live result")
    args = parser.parse_args()

    main()
