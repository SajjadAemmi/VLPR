#include <iostream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


using namespace std;


class PlateRecognizer
{
public:
    PlateRecognizer();
    ~PlateRecognizer();
    bool recognize(Plate& plate);

private:
    cv::dnn::Net net;
    cv::Mat blob;
    cv::Scalar mean;
    cv::Scalar std;
    double scalefactor;
    int plate_width;
    int plate_height;
    cv::Mat outs;
    clock_t start_time; 
    clock_t end_time;
    double score;
    double score_threshold;
    string alphabet;

    void resize();
    void preProcess();
    string postProcess(cv::Mat input);

};
