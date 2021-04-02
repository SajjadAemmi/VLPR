#include <iostream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;


class Detector
{
public:
    Detector();
    ~Detector();
    void detect(cv::Mat& image, vector<Plate>& plates);

private:
    cv::dnn::Net net;
    vector<string> outNames;
    clock_t start_time; 
    clock_t end_time;
    float score_threshold;
    float nms_threshold;
    int max_side_resize;

    void resize(cv::Mat image, int &resize_w, int &resize_h);
    void preProcess();
    void postProcess(cv::Mat& bboxes_raw, cv::Mat& scores_raw, vector<cv::RotatedRect>& bboxes, vector<float>& scores);

};