#include <iostream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "plate.h"
#include "detector.h"
#include "recognizer.h"

using namespace std;


int main(int argc, char *argv[]) 
{
  Detector detector;
  Recognizer recognizer;
  vector<Plate> plates;
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

  detector.detect(image, plates);

  cv::Point2f vertices[4];
  for (int i = 0; i < plates.size(); i++)
  {
      plates[i].rotated_rect.points(vertices);
      for (int k = 0; k < 4; k++)
          cv::line(image, vertices[k], vertices[(k+1)%4], cv::Scalar(0,255,0), 2);

      // cv::circle(image, centers[indices[i]], 4, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
      recognizer.recognize(plates[i].image);
  }

  cv::imwrite("output/9.jpg", image);

  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);
  cv::waitKey(0);

  return 0;
}
