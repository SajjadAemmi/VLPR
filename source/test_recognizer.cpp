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

#include "recognizer.cpp"

using namespace std;


int main(int argc, char *argv[]) 
{
  Recognizer recognizer;
  cv::Mat image;
  string plate_text;
  
  image = cv::imread("input/p10.jpg");
  if (!image.data)
  {
    cout << "No image data \n";
    return -1;
  }
  // cv::resize(image, image, cv::Size(1024, 768));
  // cout << image.size << endl;
  // cout << image.cols << endl;
  // cout << image.rows << endl;

  recognizer.recognize(image, &plate_text);

  // cv::imwrite("output/9.jpg", image);
  cout << plate_text;
  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);
  cv::waitKey(0);

  return 0;
}
