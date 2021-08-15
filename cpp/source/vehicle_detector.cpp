#include <iostream>

#include "vehicle_detector.h"

using namespace std;


VehicleDetector::VehicleDetector()
{
    net = cv::dnn::readNet("models/yolov3.weights", "models/yolov3.cfg");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    scalefactor = 1 / 255.0;
    mean = cv::Scalar(127.5, 127.5, 127.5);
    conf_threshold = 0.9;
    nms_threshold = 0.4;

    model = cv::dnn::DetectionModel(net);
    model.setInputParams(scalefactor, cv::Size(416, 416), mean, true);
}

VehicleDetector::~VehicleDetector()
{
    // delete ui;
}

void VehicleDetector::detect(cv::Mat& image, vector<Vehicle>& vehicles)
{
    bboxes.clear();
    centers.clear();
    class_ids.clear();
    confidences.clear();

    start_time = clock();

    model.detect(image, class_ids, confidences, bboxes, conf_threshold, nms_threshold);
    
    end_time = clock();
    cout << "vehicle detect time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;

    for (int i = 0; i < bboxes.size(); i++)
    {
        vehicles.push_back(Vehicle(image, bboxes[i], class_ids[i]));
    }
}
