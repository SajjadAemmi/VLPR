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

#include "plate.h"

using namespace std;


class PlateTracker
{
public:
    PlateTracker();
    ~PlateTracker();
    void track(cv::Mat& image, vector<Plate>& plates);
    vector<Plate> plates;

private:
    int number;
    float scalefactor;
    vector<cv::Ptr<cv::Tracker>> trackers;

    vector<cv::Point2f> centers;
    vector<float> scores;

    clock_t start_time; 
    clock_t end_time;
    float iou_threshold;
    int not_found_threshold;
};
