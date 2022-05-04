import sys
import os
from functools import partial
from datetime import datetime
import time

from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QProgressBar, QTableWidgetItem, QMainWindow, QFileDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QThread, Qt, QLine
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6 import QtGui
import cv2
from PIL.ImageQt import ImageQt
from PIL import Image

from source.process import Process


# Convert an opencv image to QPixmap
def convertCvImage2QtImage(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = cv_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)

    # rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    # PIL_image = Image.fromarray(rgb_image).convert('RGB')
    # return QPixmap.fromImage(ImageQt(PIL_image))


class Main(QMainWindow):
    def __init__(self):
        super(Main, self).__init__()
        loader = QUiLoader()
        self.ui = loader.load('ui/mainwindow.ui')
        self.ui.show()

        self.ui.btn_browse.clicked.connect(self.browse)
        self.ui.btn_play.clicked.connect(self.play)
        self.ui.btn_pause.clicked.connect(self.pause)
        self.ui.btn_stop.clicked.connect(self.stop)
        self.ui.btn_play.setIcon(QIcon('icons/play.ico'))
        self.ui.btn_pause.setIcon(QIcon('icons/pause.ico'))
        self.ui.btn_stop.setIcon(QIcon('icons/stop.ico'))
        self.ui.btn_direction_up.setIcon(QIcon('icons/arrow_up.ico'))
        self.ui.btn_direction_down.setIcon(QIcon('icons/arrow_down.ico'))

        self.lbl_class_ids = []
        for i in range(5):
            self.lbl_class_ids.append(self.ui.findChild(QLabel, f"lbl_class_{i}"))

        # self.length_stw = 0
        self.process = Process()
        self.process.signalShowPreview.connect(self.slotShowPreview)
        self.process.signalShowPlate.connect(self.slotShowPlate)
        self.process.signalUpdateClassCounters.connect(self.slotUpdateClassCounters)
        self.index_plate = 0

    def browse(self):
        self.input_file_path = QFileDialog.getOpenFileName(self, 'Select file', dir='./input', options=QFileDialog.DontUseNativeDialog)[0]
        self.ui.tb_path.setText(self.input_file_path)

    def play(self):
        self.process.setInputFilePath(self.input_file_path)
        self.process.start()

    def pause(self):
        pass

    def stop(self):
        pass

    def slotShowPreview(self, image):
        pixmap = convertCvImage2QtImage(image)
        self.ui.lbl_preview.setPixmap(pixmap)

    def slotShowPlate(self, plate):
        self.ui.tableWidget.insertRow(self.ui.tableWidget.rowCount())

        index_last_row = self.ui.tableWidget.rowCount() - 1

        pixmap = convertCvImage2QtImage(plate.image)
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)
        self.ui.tableWidget.setCellWidget(index_last_row, 1, image_label)

        textItem = QTableWidgetItem()
        textItem.setText(plate.text)
        self.ui.tableWidget.setItem(index_last_row, 0, textItem)

        self.ui.tableWidget.resizeRowsToContents()

    def slotUpdateClassCounters(self, class_id):
        if class_id < 5:
            count = int(self.lbl_class_ids[class_id].text())
            count += 1
            self.lbl_class_ids[class_id].setText(str(count))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Main()
    sys.exit(app.exec())
