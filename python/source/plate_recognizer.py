import re
from .functions import *
from .plate import Plate


class PlateRecognizer:
    alphabet = "#0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"

    def __init__(self):
        self.net = cv2.dnn.readNet("models/plate_recognizer.onnx")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.mean = (123.68, 116.78, 103.94)
        self.scale_factor = 1.0
        self.threshold = 0.9

    def recognize(self, plate):
        try:
            image = cv2.cvtColor(plate.image, cv2.COLOR_RGB2BGR)

            blob = cv2.dnn.blobFromImage(image, size=(320, 64), mean=self.mean, scalefactor=self.scale_factor)
            self.net.setInput(blob)

            # Run the recognition model
            result = self.net.forward()

            softmax_result = self.softmax(result)
            values = np.max(softmax_result, axis=2).flatten()
            prob = np.argmax(softmax_result, axis=2).flatten()

            preds_idx = np.nonzero(prob)
            confidence = np.mean(values[preds_idx])

            # decode the result into text
            text = self.decodeText(result)

            # find char index in text
            find_index = re.search(r"[a-zA-Z]", text, re.I)
            if find_index is not None:
                i = find_index.start()
                if len(text) >= 8 and confidence >= self.threshold:
                    plate.text = text[i-2:i] + ' ' + alphabetDict(text[i]) + ' ' + text[i+1:i+4] + ' - ' + text[i+4:i+6]
                    print(f'OCR process done for plate id {plate.id}, confidence: {confidence}')

        except Exception as e:
            print(f'error in recognize function:', e)

    def decodeText(self, scores):
        text = ""
        for i in range(scores.shape[0]):
            c = np.argmax(scores[i][0])
            if c != 0:
                text += self.alphabet[c - 1]
            else:
                text += '-'

        # adjacent same letters as well as background text must be removed to get the final output
        char_list = []
        for i in range(len(text)):
            # print(text)
            if text[i] != '-' and not (i > 0 and text[i] == text[i - 1]):
                char_list.append(text[i])

        return ''.join(char_list)

    def softmax(self, a):
        """Compute softmax values for each sets of scores in x."""
        result = []
        for x in a:
            e_x = np.exp(x - np.max(x))
            result.append(e_x / e_x.sum(axis=1))

        return np.array(result)
