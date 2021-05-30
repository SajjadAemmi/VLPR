#include <iostream>

#include "plate.h"

using namespace std;


cv::Size Plate::outputSize = cv::Size(320, 64);
cv::Point2f Plate::dst_corners[4] = {cv::Point2f(0, outputSize.height - 1),
                                     cv::Point2f(0, 0),
                                     cv::Point2f(outputSize.width - 1, 0),
                                     cv::Point2f(outputSize.width - 1, outputSize.height - 1)};

Plate::Plate(cv::Mat& image, cv::RotatedRect rotated_rect, int id = 0)
{
    this->rotated_rect = rotated_rect;
    this->roi = rotated_rect.boundingRect();
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
    cv::Point2f roi_corners[4];
    this->rotated_rect.points(roi_corners);

    cv::Mat rotationMatrix = cv::getPerspectiveTransform(roi_corners, dst_corners);
    cv::warpPerspective(frame, this->image, rotationMatrix, outputSize);
}
