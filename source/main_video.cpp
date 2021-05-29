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
  cv::Mat frame;
  cv::Point2f vertices[4];
  clock_t start_time; 
  clock_t end_time;

  cv::VideoCapture cap("input/C0012.MP4"); 
  if(!cap.isOpened())
  {
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  
  // VideoWriter video("outcpp.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height));
	
  while(true)
  {
    start_time = clock();

    plates.clear();

    cap >> frame; 
    if (frame.empty())
      break;

    detector.detect(frame, plates);

    for (int i = 0; i < plates.size(); i++)
    {
        plates[i].rotated_rect.points(vertices);
        for (int k = 0; k < 4; k++)
            cv::line(frame, vertices[k], vertices[(k+1)%4], cv::Scalar(0,255,0), 2);
        // cv::circle(image, centers[indices[i]], 4, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);

        recognizer.recognize(plates[i].image);
    }

    cv::imshow("Frame", frame );
    // video.write(frame);

    // Press ESC on keyboard to exit
    if((char)cv::waitKey(1) == 27)
      break;
    
    end_time = clock();
    cout << "total process frame time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;
  }
 
  // When everything done, release the video capture object
  cap.release();
  // video.release();

  // Closes all the frames
  cv::destroyAllWindows();

  // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);

  return 0;
}
