#include <iostream>
#include <time.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>

#include "plate_detector.cpp"
#include "plate.cpp"

using namespace std;


int main(int argc, char *argv[]) 
{
  PlateDetector plate_detector;
  cv::Mat image;
  vector<Plate> plates;
  
  image = cv::imread("input/6.bmp");
  if (!image.data)
  {
    cout << "No image data \n";
    return -1;
  }
  cout << image.size << "\t" << image.cols << "\t" << image.rows << endl;
  
  plate_detector.detect(image, plates);

  cv::resize(image, image, cv::Size(1280, 960));
  for (int i = 0; i < plates.size(); i++)
  {
      cv::circle(image, plates[i].rotated_rect.center, 4, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
      for (int k = 0; k < 4; k++)
      {
          cv::line(image, plates[i].roi[k], plates[i].roi[(k+1)%4], cv::Scalar(0,255,0), 2);
      }
  }

  cv::imwrite("output/6.jpg", image);
  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  // cv::imshow("Display Image", image);
  // cv::waitKey(0);

  return 0;
}
