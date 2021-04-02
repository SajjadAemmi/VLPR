#include <iostream>

#include "plate.h"

using namespace std;


Plate::Plate(cv::Mat image, cv::RotatedRect rotated_rect)
{
    this->rotated_rect = rotated_rect;
    this->rect = rotated_rect.boundingRect();
    this->image = image(rect);
}

Plate::~Plate()
{
    // delete ui;
}

void Plate::detect(cv::Mat& image)
{

}