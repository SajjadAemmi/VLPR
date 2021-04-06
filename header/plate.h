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
public:
    Plate(cv::Mat image, cv::RotatedRect rotated_rect);
    ~Plate();
    void detect(cv::Mat& image);

    cv::RotatedRect rotated_rect;
    cv::Rect rect;
    cv::Mat image;
    string text;
};