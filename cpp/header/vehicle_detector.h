#include <iostream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "vehicle.h"

using namespace std;


class VehicleDetector
{
public:
    VehicleDetector();
    ~VehicleDetector();
    void detect(cv::Mat& image, vector<Vehicle>& vehicles);

private:
    cv::dnn::Net net;
    cv::dnn::DetectionModel model;

    cv::Scalar mean;
    float scalefactor;
    vector<string> outNames;
    vector<cv::Rect> bboxes;
    vector<cv::Point2f> centers;
    vector<float> confidences;
    vector<int> class_ids;
    int w;
    int h;

    clock_t start_time; 
    clock_t end_time;
    float conf_threshold;
    float nms_threshold;

    void resize(cv::Mat image, int &resize_w, int &resize_h);
    void preProcess();
    void postProcess(cv::Mat &outs, vector<cv::Rect> &bboxes, vector<float> &scores);
};