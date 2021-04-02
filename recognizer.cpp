#include <iostream>

#include "recognizer.h"

using namespace std;


Recognizer::Recognizer()
{
    net = cv::dnn::readNet("model/detector.pb");
    outNames = {"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"};
    score_threshold = 0.95;
    nms_threshold = 0.2;
}

Recognizer::~Recognizer()
{
    // delete ui;
}

void Recognizer::detect(cv::Mat& image)
{
    start_time = clock();
}