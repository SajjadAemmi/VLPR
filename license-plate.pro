#-------------------------------------------------
#
# Project created by QtCreator 2019-08-20T17:46:56
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = license-plate
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

QMAKE_LFLAGS += -no-pie

CONFIG += \
        c++17

HEADERS += \
        header/mainwindow.h \
        header/process.h \
        header/plate.h \
        header/plate_detector.h \
        header/plate_tracker.h \
        header/vehicle.h \
#        header/vehicle_detector.h \
#        header/vehicle_tracker.h \
        header/recognizer.h

SOURCES += \
        source/main_video_gui.cpp \
        source/mainwindow.cpp \
        source/process.cpp \
        source/plate.cpp \
        source/plate_detector.cpp \
        source/plate_tracker.cpp \
        source/vehicle.cpp \
#        source/vehicle_detector.cpp \
#        source/vehicle_tracker.cpp \
        source/recognizer.cpp

FORMS += \
        ui/mainwindow.ui

RESOURCES += \
        icon/play.ico \
        icon/pause.ico \
        icon/stop.ico

INCLUDEPATH += /usr/local/include/opencv4 \
                header \
                source

LIBS += -L/usr/local/lib \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_shape \
        -lopencv_videoio \
        -lopencv_dnn \
        -lopencv_video \
        -lopencv_tracking \
#        -lopencv_features2d \
#        -lopencv_calib3d \
#        -lopencv_objdetect \
#        -lopencv_stitching

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
