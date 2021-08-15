#include <iostream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "vehicle.h"

using namespace std;


class VehicleTracker
{
public:
    VehicleTracker();
    ~VehicleTracker();
    void track(cv::Mat& image, vector<Vehicle>& vehicles);
    vector<cv::Rect> rois;
    vector<Vehicle> vehicles;

private:
    cv::dnn::Net net;
    cv::Scalar mean;
    float scalefactor;
    vector<cv::Ptr<cv::Tracker>> trackers;

    vector<cv::Point2f> centers;
    vector<float> scores;

    clock_t start_time; 
    clock_t end_time;
    float iou_threshold;
};
