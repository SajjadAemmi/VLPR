#include <iostream>

#include "detector.h"

using namespace std;


Detector::Detector()
{
    net = cv::dnn::readNet("model/detector.pb");
    outNames = {"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"};
    score_threshold = 0.95;
    nms_threshold = 0.2;
}

Detector::~Detector()
{
    // delete ui;
}

void Detector::detect(cv::Mat& image)
{
    start_time = clock();

    vector<cv::RotatedRect> bboxes;
    vector<cv::Point2f> centers;
    vector<float> scores;

    int input_col_size = 1024;
    int input_row_size = 768;

    cv::Mat blob;
    cv::Scalar mean = cv::Scalar(123.68, 116.78, 103.94);
    cv::dnn::blobFromImage(image, blob, 1.0, cv::Size(input_col_size, input_row_size), mean, true, false); // Create a 4D blob from a frame.
    net.setInput(blob, "");

    vector<cv::Mat> outs;
    net.forward(outs, outNames);

    cv::Mat scores_raw = outs[0];
    cv::Mat bboxes_raw = outs[1];

    // cout << scores_raw.size << endl;
    // cout << bboxes_raw.size << endl;
    
    scores_raw = scores_raw.reshape (1, vector<int> {1, bboxes_raw.size[2], bboxes_raw.size[3]});
    bboxes_raw = bboxes_raw.reshape (1, vector<int> {5, bboxes_raw.size[2], bboxes_raw.size[3]});

    // cout << scores_raw.size << endl;
    // cout << bboxes_raw.size << endl;

    cv::resize(image, image, cv::Size(input_col_size, input_row_size));

    postProcess(bboxes_raw, scores_raw, bboxes, scores);

    // nms
    vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, score_threshold, nms_threshold,	indices);

    cv::Point2f vertices[4];
    for (int i = 0; i < indices.size(); i++)
    {
        bboxes[indices[i]].points(vertices);
        for (int k = 0; k < 4; k++)
            cv::line(image, vertices[k], vertices[(k+1)%4], cv::Scalar(0,255,0), 2);

        // cv::circle(image, centers[indices[i]], 4, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
    }

    end_time = clock();
    cout << float(end_time - start_time) / CLOCKS_PER_SEC << " seconds" << endl;
}

void Detector::preProcess()
{

} 

void Detector::postProcess(cv::Mat& bboxes_raw, cv::Mat& scores_raw, vector<cv::RotatedRect>& bboxes, vector<float>& scores)
{
    int r_min, r_max, c_min, c_max, w, h;
    float angle;

    for (int i = 0; i < bboxes_raw.size[1]; i++)
    {
        for (int j = 0; j < bboxes_raw.size[2]; j++)
        {              
        // if(scores_raw.at<float>(0, i, j) > 0.95)
        // {
            scores.push_back(scores_raw.at<float>(0, i, j));

            r_min = i*4 - bboxes_raw.at<float>(0, i, j);
            c_max = j*4 + bboxes_raw.at<float>(1, i, j);
            r_max = i*4 + bboxes_raw.at<float>(2, i, j);
            c_min = j*4 - bboxes_raw.at<float>(3, i, j);
            w = c_max - c_min;
            h = r_max - r_min; 
            angle = bboxes_raw.at<float>(4, i, j) * -180 / M_PI;
            
            bboxes.push_back(cv::RotatedRect(cv::Point2f(c_min + w/2, r_min + h/2), cv::Size2f(w, h), angle));
            // centers.push_back(cv::Point2f(j*4, i*4));

            // Rect brect = rRect.boundingRect();
            // rectangle(test_image, brect, Scalar(255,0,0), 2);
        // }
        }
    }
}