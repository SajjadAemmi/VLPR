#include <iostream>

#include "plate.h"
#include "detector.h"

using namespace std;


Detector::Detector()
{
    net = cv::dnn::readNet("model/detector.pb");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    outNames = {"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"};
    mean = cv::Scalar(123.68, 116.78, 103.94);
    scalefactor = 1.0;
    score_threshold = 0.95;
    nms_threshold = 0.2;
    max_side_resize = 2400;
}

Detector::~Detector()
{
    // delete ui;
}

void Detector::detect(cv::Mat &image, vector<Plate> &plates)
{
    bboxes.clear();
    centers.clear();
    scores.clear();
    outs.clear();

    int resize_w, resize_h;
    resize(image, resize_w, resize_h);

    start_time = clock();
    
    cv::dnn::blobFromImage(image, blob, scalefactor, cv::Size(resize_w, resize_h), mean, true, false); // Create a 4D blob from a frame.
    net.setInput(blob, "");
    net.forward(outs, outNames);

    end_time = clock();
    cout << "detect forward time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;

    scores_raw = outs[0];
    bboxes_raw = outs[1];

    // cout << scores_raw.size << endl;
    // cout << bboxes_raw.size << endl;
    
    scores_raw = scores_raw.reshape(1, vector<int> {1, scores_raw.size[2], scores_raw.size[3]});
    bboxes_raw = bboxes_raw.reshape(1, vector<int> {5, bboxes_raw.size[2], bboxes_raw.size[3]});

    // cout << scores_raw.size << endl;
    // cout << bboxes_raw.size << endl;

    start_time = clock();
    
    cv::resize(image, image, cv::Size((int)resize_w, (int)resize_h));
    postProcess(bboxes_raw, scores_raw, bboxes, scores);

    end_time = clock();
    cout << "postProcess time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;

    start_time = clock();
    
    // nms
    vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, score_threshold, nms_threshold, indices);

    end_time = clock();
    cout << "nms time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;

    for (int i = 0; i < indices.size(); i++)
        plates.push_back(Plate(image, bboxes[indices[i]]));

}

// resize image to a size multiple of 32 which is required by the network
// param max_side_len: limit of max image size to avoid out of memory in gpu
// return: the resized image and the resize ratio
void Detector::resize(cv::Mat image, int &resize_w, int &resize_h)
{
    float ratio;
    resize_w = image.cols;
    resize_h = image.rows;
    
    // limit the max side
    if(max(resize_h, resize_w) > max_side_resize)
        if(resize_h > resize_w)
            ratio = float(max_side_resize) / resize_h;
        else
            ratio = float(max_side_resize) / resize_w;
    else
        ratio = 1;
        
    resize_h = (int)resize_h * ratio;
    resize_w = (int)resize_w * ratio;

    if(resize_h % 32 != 0)
        resize_h = (resize_h / 32 - 1) * 32;

    if(resize_w % 32 != 0)
        resize_w = (resize_w / 32 - 1) * 32;
}

void Detector::preProcess()
{

} 

void Detector::postProcess(cv::Mat &bboxes_raw, cv::Mat &scores_raw, vector<cv::RotatedRect> &bboxes, vector<float> &scores)
{
    int r_min, r_max, c_min, c_max, w, h;
    float angle;

    for (int i = 0; i < bboxes_raw.size[1]; i++)
    {
        for (int j = 0; j < bboxes_raw.size[2]; j++)
        {              
            if(scores_raw.at<float>(0, i, j) > score_threshold)
            {
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
                // rectangle(test_image, brect, Scalar(255,0,0), 2);
            }
        }
    }
}
