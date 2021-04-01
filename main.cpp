#include <iostream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "detector.h"

using namespace std;


int main(int argc, char *argv[]) 
{
  Detector detector; 
  cv::Mat image;
  
  image = cv::imread("input/9.jpg");
  if (!image.data)
  {
    cout << "No image data \n";
    return -1;
  }
  // cout << image.size << endl;
  // cout << image.cols << endl;
  // cout << image.rows << endl;

  detector.detect(image);

  cv::imwrite("output/9.jpg", image);

  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);
  cv::waitKey(0);

  return 0;
}

// cmake -S . -B build -DCMAKE_OSX_ARCHITECTURES=arm64
// cmake --build build
