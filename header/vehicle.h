#ifndef __VEHICLE_H__
#define __VEHICLE_H__

#include <iostream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;


class Vehicle
{
public:
    Vehicle(cv::Mat image, cv::Rect rect, int class_id);
    ~Vehicle();
    void detect(cv::Mat& image);

    cv::Rect roi;
    cv::Mat image;
    int class_id;
    int not_found;
};

#endif
