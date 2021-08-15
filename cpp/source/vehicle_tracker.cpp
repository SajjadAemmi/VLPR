#include <iostream>

#include "vehicle_tracker.h"

using namespace std;


VehicleTracker::VehicleTracker()
{
    iou_threshold = 0.1;
}

VehicleTracker::~VehicleTracker()
{
    // delete ui;
}

void VehicleTracker::track(cv::Mat& image, vector<Vehicle>& vehicles)
{
//    centers.clear();
//    scores.clear();

    start_time = clock();

    for (int i = 0; i < this->trackers.size(); i++)
    {
        trackers[i]->update(image, rois[i]);
    }

    bool same_flag;
    for (int i = 0; i < vehicles.size(); i++)
    {
        same_flag = false;
        for (int j = 0; j < this->trackers.size(); j++)
        {
            cv::Rect rects_intersection = vehicles[i].roi & rois[j];
            cv::Rect rects_union = vehicles[i].roi | rois[j];

            float iou = (float)rects_intersection.area() / (float)rects_union.area();
            cout << "iou" << iou << endl;
            if(iou > iou_threshold)
            {
                same_flag = true;
                break;
            }
        }
        if(same_flag == false)
        {
            rois.push_back(vehicles[i].roi);
//            cv::Ptr<cv::Tracker> t = cv::TrackerGOTURN::create();
            cv::Ptr<cv::Tracker> t = cv::TrackerCSRT::create();
            t->init(image, vehicles[i].roi);
            trackers.push_back(t);
        }
    }

//    for (int i = 0; i < this->trackers.size(); i++)
//    {
//        same_flag = false;
//        for (int j = 0; j < vehicles.size(); j++)
//        {
//            cv::Rect rects_intersection = vehicles[j].roi & rois[i];
//            cv::Rect rects_union = vehicles[j].roi | rois[i];

//            float iou = (float)rects_intersection.area() / (float)rects_union.area();
//            cout << "iou" << iou << endl;
//            if(iou > iou_threshold)
//            {
//                same_flag = true;
//                break;
//            }
//        }
//        if(same_flag == false)
//        {
//            rois.push_back(vehicles[i].roi);
//            trackers.push_back(t);
//        }
//    }

    end_time = clock();
    cout << "track time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;
}
