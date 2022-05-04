#include <iostream>

#include "plate_tracker.h"

using namespace std;


PlateTracker::PlateTracker()
{
    iou_threshold = 0.1;
    not_found_threshold = 10;
    number = 0;
}

PlateTracker::~PlateTracker()
{
    // delete ui;
}

void PlateTracker::track(cv::Mat& frame, vector<Plate>& plates)
{
    start_time = clock();

    for (int i = 0; i < this->trackers.size(); i++)
    {
        trackers[i]->update(frame, this->plates[i].roi);
        this->plates[i].not_found++;
    }

    bool same_flag;
    for (int i = 0; i < plates.size(); i++)
    {
        same_flag = false;
        for (int j = 0; j < this->trackers.size(); j++)
        {
            cv::Rect rects_intersection = plates[i].roi & this->plates[j].roi;
            cv::Rect rects_union = plates[i].roi | this->plates[j].roi;
            float iou = (float)rects_intersection.area() / (float)rects_union.area();
            if(iou > iou_threshold)
            {
                this->trackers[j]->init(frame, plates[i].roi);
                this->plates[j].rotated_rect = plates[i].rotated_rect;
                this->plates[j].roi = plates[i].roi;
                this->plates[j].perspectiveTransform(frame);
                this->plates[j].not_found = 0;
                same_flag = true;
                break;
            }
        }
        if(same_flag == false)
        {
            this->plates.push_back(Plate(frame, plates[i].rotated_rect, this->number));
//            cv::Ptr<cv::Tracker> t = cv::TrackerGOTURN::create();
            cv::Ptr<cv::Tracker> t = cv::TrackerCSRT::create();
            t->init(frame, plates[i].roi);
            trackers.push_back(t);
            this->number++;
        }
    }

    for (int i = this->trackers.size() - 1; i >= 0; i--)
    {
        if(this->plates[i].not_found >= not_found_threshold || this->plates[i].roi.y >= frame.rows * 6 / 7)
        {
            this->plates.erase(this->plates.begin() + i);
            this->trackers.erase(this->trackers.begin() + i);
        }
    }

    end_time = clock();
//    cout << "track time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;
}
