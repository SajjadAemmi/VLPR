#include <iostream>

#include "recognizer.h"

using namespace std;


Recognizer::Recognizer()
{
    net = cv::dnn::readNet("model/recognizer.onnx");
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
