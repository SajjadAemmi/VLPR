#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <QMainWindow>
#include "QLineEdit"
#include "QThread"
#include "ui_mainwindow.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "plate_detector.h"
#include "plate_tracker.h"
//#include "vehicle_detector.h"
//#include "vehicle_tracker.h"
#include "recognizer.h"

using namespace std;


class Process : public QThread
{
    Q_OBJECT

public:
    explicit Process();
    ~Process();
    void run();
    string video_path_str;

private:
    clock_t start_time;
    clock_t end_time;
    PlateDetector plate_detector;
    PlateTracker plate_tracker;
//    VehicleDetector vehicle_detector;
//    VehicleTracker vehicle_tracker;
    Recognizer recognizer;
    cv::VideoCapture cap;
    cv::Scalar color;
    float process_boundary_line_y;

signals:
   void signalShowPreview(cv::Mat);
   void signalShowPlate(cv::Mat, QString);
   void signalUpdateClassCounters(int class_id);

public slots:
   void slotChangeSlidervalue(int);
};
