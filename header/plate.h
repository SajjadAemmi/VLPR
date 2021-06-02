#ifndef __PLATE_H__
#define __PLATE_H__

#include <iostream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;


class Plate
{
private:
    static cv::Size outputSize;
    static cv::Point2f dst_corners[4];

public:
    Plate(cv::Mat& image, cv::RotatedRect rotated_rect, vector<cv::Point2f> roi, int id);
    ~Plate();
    void updateImage(cv::Mat image);
    void perspectiveTransform(cv::Mat& image);

    cv::RotatedRect rotated_rect;
    cv::Rect rect;
    vector<cv::Point2f> roi;
    cv::Mat image;
    string text;
    int not_found;
    int id;
    bool recognized;
};

#endif
