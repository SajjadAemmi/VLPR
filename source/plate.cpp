#include <iostream>

#include "plate.h"

using namespace std;


cv::Size Plate::outputSize = cv::Size(320, 64);
cv::Point2f Plate::dst_corners[4] = {cv::Point2f(0, outputSize.height - 1),
                                     cv::Point2f(0, 0),
                                     cv::Point2f(outputSize.width - 1, 0),
                                     cv::Point2f(outputSize.width - 1, outputSize.height - 1)};

Plate::Plate(cv::Mat& image, cv::RotatedRect rotated_rect, vector<cv::Point2f> roi, int id = 0)
{
    this->rotated_rect = rotated_rect;
    this->rect = rotated_rect.boundingRect();
    this->roi = roi;
    this->id = id;
    this->not_found = 0;
    this->recognized = false;

    this->perspectiveTransform(image);
}

Plate::~Plate()
{
    // delete ui;
}

void Plate::perspectiveTransform(cv::Mat& frame)
{
    cv::Point2f* a = &roi[0];
    cv::Mat rotationMatrix = cv::getPerspectiveTransform(a, dst_corners);
    cv::warpPerspective(frame, this->image, rotationMatrix, outputSize);
}
