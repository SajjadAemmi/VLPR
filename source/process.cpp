#include <iostream>
#include "time.h"
#include "vector"
#include "QFile"
#include "QDir"
#include "QtDebug"
#include "QMessageBox"
#include "QFileDialog"
#include "QTextStream"

#include "process.h"

using namespace std;

Process::Process()
{
    color = cv::Scalar(255, 255, 255);
    process_boundary_line_y = 960.0 * 80.0 / 100.0;
}

Process::~Process()
{

}

void Process::run()
{
//    vector<Vehicle> vehicles;
    vector<Plate> plates;
    cv::Mat frame, frame_preview;
    cv::Point2f vertices[4];
    clock_t start_time;
    clock_t end_time;
    string plate_text;

    cap = cv::VideoCapture(video_path_str);
    if(!cap.isOpened())
    {
      qDebug() << "Error opening video stream or file" << endl;
      return;
    }

    // VideoWriter video("outcpp.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height));

    while(true)
    {
      start_time = clock();

      plates.clear();
//      vehicles.clear();

      cap >> frame;
      if (frame.empty())
        break;

//      cv::resize(frame, frame, cv::Size(), 0.5, 0.5);

      plate_detector.detect(frame, plates);      
      frame_preview = frame.clone();
      for (int i = 0; i < plates.size(); i++)
      {
          plates[i].rotated_rect.points(vertices);

          for (int k = 0; k < 4; k++)
              cv::line(frame_preview, vertices[k], vertices[(k+1)%4], cv::Scalar(0,255,0), 2);
          // cv::circle(image, centers[indices[i]], 4, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
      }

      plate_tracker.track(frame, plates);
      for (int i = 0; i < plate_tracker.plates.size(); i++)
      {
//          cv::rectangle(frame_preview, plate_tracker.plates[i].roi, cv::Scalar(255, 0, 0));
//          cv::putText(frame_preview, to_string(plate_tracker.plates[i].id), cv::Point(plate_tracker.plates[i].roi.x, plate_tracker.plates[i].roi.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

          if(plate_tracker.plates[i].roi.y >= process_boundary_line_y && plate_tracker.plates[i].recognized == false)
          {
              if(recognizer.recognize(plate_tracker.plates[i].image, &plate_text))
              {
                  cout << "text: " << plate_text << endl;
                  plate_tracker.plates[i].recognized = true;
                  emit signalShowPlate(plate_tracker.plates[i].image, QString::fromStdString(plate_text));
              }
          }
      }

      // process boundary line
      cv::line(frame_preview, cv::Point2f(0, process_boundary_line_y), cv::Point2f(frame.cols, process_boundary_line_y), cv::Scalar(0,0,255), 1);

//      vehicle_detector.detect(frame, vehicles);
//      for (int i = 0; i < vehicles.size(); i++)
//      {
//          cv::rectangle(frame_preview, vehicles[i].roi, cv::Scalar(0, 0, 255));
//          cout << vehicles[i].class_id << endl;
//          emit signalUpdateClassCounters(vehicles[i].class_id);
//      }

      emit signalShowPreview(frame_preview);
      // video.write(frame);

      end_time = clock();
    }

    // When everything done, release the video capture object
    cap.release();
    // Closes all the frames
    cv::destroyAllWindows();

    end_time = clock();
    qDebug() << float(end_time - start_time) / CLOCKS_PER_SEC << " seconds";
}

void Process::slotChangeSlidervalue(int value)
{
    process_boundary_line_y = 960.0 * (float)value / 100.0;
}
