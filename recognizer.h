#include <iostream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;


class Recognizer
{
public:
    Recognizer();
    ~Recognizer();
    void recognize(cv::Mat& image);

private:
    cv::dnn::Net net;
    cv::Mat blob;
    cv::Scalar mean;
    float scalefactor;
    int plate_width;
    int plate_height;
    vector<cv::Mat> outs;
    clock_t start_time; 
    clock_t end_time;
    float score_threshold;
    float nms_threshold;

    void resize();
    void preProcess();
    void postProcess(cv::Mat& bboxes_raw, cv::Mat& scores_raw, vector<cv::RotatedRect>& bboxes, vector<float>& scores);

};
