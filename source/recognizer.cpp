#include <iostream>

#include "recognizer.h"

using namespace std;


Recognizer::Recognizer()
{
    net = cv::dnn::readNet("model/recognizer.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    mean = cv::Scalar(127.5, 127.5, 127.5);
    plate_width = 320;
    plate_height = 64;
    scalefactor = 1 / 127.5;
    score_threshold = 0.95;
    nms_threshold = 0.2;
}

Recognizer::~Recognizer()
{
    // delete ui;
}

void Recognizer::recognize(cv::Mat& image)
{
    outs.clear();

    start_time = clock();

    cv::dnn::blobFromImage(image, blob, scalefactor, cv::Size(plate_width, plate_height), mean, true, false); // Create a 4D blob from a plate image.
    net.setInput(blob, "");
    net.forward(outs);

    cout << outs[0].size << endl;

    end_time = clock();
    cout << "recognize forward time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;
}
