#include <iostream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "plate.h"
#include "plate_detector.h"
#include "plate_recognizer.h"

using namespace std;


clock_t start_time, end_time;
PlateDetector plate_detector;
PlateRecognizer plate_recognizer;

void processImage(cv::Mat &image)
{
    start_time = clock();
    vector<Plate> plates;
    plate_detector.detect(image, plates);

    cv::Point2f vertices[4];
    for(int i = 0; i < plates.size(); i++)
    {
        plates[i].rotated_rect.points(vertices);
        for (int k = 0; k < 4; k++)
            cv::line(image, vertices[k], vertices[(k+1)%4], cv::Scalar(0,255,0), 2);

        // cv::circle(image, centers[indices[i]], 4, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
        plate_recognizer.recognize(plates[i]);
        cout << "text: " << plates[i].text << endl;
    }

    end_time = clock();
    cout << "total process frame time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;
    cv::imshow("Display Image", image);
}

int main(int argc, char *argv[]) 
{
    string input_file_path = argv[1];
    filesystem::path path(input_file_path);
    if(path.extension() == ".jpg")
    {
        cv::Mat image;
        image = cv::imread(input_file_path);
        if(!image.data)
        {
            cout << "No image data \n";
            return -1;
        }
        processImage(image);
        cv::imwrite("output/9.jpg", image);
        cv::waitKey(0);
    }
    else if(path.extension() == ".mp4")
    {
        cv::Mat frame;
        cv::VideoCapture cap(input_file_path);
        while(true)
        {
            cap >> frame;
            if(frame.empty())
                break;

            processImage(frame);
            char c = (char)cv::waitKey(1);
            if(c==27) // Press  ESC on keyboard to exit
                break;
        }
        cap.release();
        cv::destroyAllWindows();
    }

  // cout << image.size << endl;
  // cout << image.cols << endl;
  // cout << image.rows << endl;

  return 0;
}
