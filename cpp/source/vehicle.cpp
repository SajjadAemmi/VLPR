#include <iostream>

#include "vehicle.h"

using namespace std;


Vehicle::Vehicle(cv::Mat image, cv::Rect roi, int class_id)
{
    this->roi = roi;
    this->image = image(roi);
    this->class_id = class_id;
    this->not_found = 0;
}

Vehicle::~Vehicle()
{
    // delete ui;
}

void Vehicle::detect(cv::Mat& image)
{

}
