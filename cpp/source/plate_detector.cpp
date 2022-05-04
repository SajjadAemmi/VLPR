#include <iostream>

#include "plate_detector.h"

using namespace std;


PlateDetector::PlateDetector()
{
    net = cv::dnn::readNet("models/plate_detector.pb");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    outNames = {"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"};
    mean = cv::Scalar(123.68, 116.78, 103.94);
    scalefactor = 1.0;
    score_threshold = 0.95;
    nms_threshold = 0.2;
    max_side_resize = 2400;
}

PlateDetector::~PlateDetector()
{
    // delete ui;
}

void PlateDetector::detect(cv::Mat& image, vector<Plate>& plates)
{
    bboxes.clear();
    rois.clear();
    scores.clear();
    outs.clear();

    int resize_w = 1280, resize_h = 960;
    // resize(image, resize_w, resize_h);

//    start_time = clock();
    cv::dnn::blobFromImage(image, blob, scalefactor, cv::Size(resize_w, resize_h), mean, true, false); // Create a 4D blob from a frame.
    net.setInput(blob);
    net.forward(outs, outNames);
//    end_time = clock();
//    cout << "detect forward time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;

    scores_raw = outs[0].reshape(1, vector<int> {1, outs[0].size[2], outs[0].size[3]});
    bboxes_raw = outs[1].reshape(1, vector<int> {5, outs[1].size[2], outs[1].size[3]});
    // cout << scores_raw.size << endl;
    // cout << bboxes_raw.size << endl;

//    start_time = clock();
    cv::resize(image, image, cv::Size((int)resize_w, (int)resize_h));
    postProcess(bboxes_raw, scores_raw);
//    end_time = clock();
//    cout << "postProcess time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;

    // nms
//    start_time = clock();
    vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, score_threshold, nms_threshold, indices, 1.f, 0);
//    end_time = clock();
//    cout << "nms time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;

    for (int i = 0; i < indices.size(); i++)
        plates.push_back(Plate(image, bboxes[indices[i]], rois[indices[i]], -1));

    // for (int i = 0; i < bboxes.size(); i++)
    // {
    //     plates.push_back(Plate(image, bboxes[i], -1));
    // }
}

// resize image to a size multiple of 32 which is required by the network
// param max_side_len: limit of max image size to avoid out of memory in gpu
// return: the resized image and the resize ratio
void PlateDetector::resize(cv::Mat image, int &resize_w, int &resize_h)
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

void PlateDetector::preProcess()
{

}

void PlateDetector::postProcess(cv::Mat &geo, cv::Mat &scores_raw)
{
    float d0, d1, d2, d3, w, h, angle, x, y;

    for (float i = 0; i < bboxes_raw.size[1]; i++)
    {
        for (float j = 0; j < bboxes_raw.size[2]; j++)
        {              
            if(scores_raw.at<float>(0, i, j) > score_threshold)
            {
                scores.push_back(scores_raw.at<float>(0, i, j));

                d0 = geo.at<float>(0, i, j);
                d1 = geo.at<float>(1, i, j);
                d2 = geo.at<float>(2, i, j);
                d3 = geo.at<float>(3, i, j);
                angle = geo.at<float>(4, i, j);

                float origin_array[2] = {j*4, i*4};
                cv::Mat origin = cv::Mat(1, 2, CV_32F, origin_array);

                float data[10] = {0, -d0 - d2, d1 + d3, -d0 - d2, d1 + d3, 0, 0, 0, d3, -d2};
                cv::Mat p = cv::Mat(5, 2, CV_32F, data);

                float *rotate_matrix_x_array;
                float *rotate_matrix_y_array;
                if(angle > 0)
                {
                    rotate_matrix_x_array = new float[2] {cos(angle), sin(angle)};
                    rotate_matrix_y_array = new float[2] {-sin(angle), cos(angle)};
                }
                else
                {
                    rotate_matrix_x_array = new float[2]{cos(-angle), -sin(-angle)};
                    rotate_matrix_y_array = new float[2]{sin(-angle), cos(-angle)};
                }

                cv::Mat rotate_matrix_x = cv::Mat(2, 1, CV_32F, rotate_matrix_x_array);
                rotate_matrix_x = cv::repeat(rotate_matrix_x, 1, 5).t();

                cv::Mat rotate_matrix_y = cv::Mat(2, 1, CV_32F, rotate_matrix_y_array);
                rotate_matrix_y = cv::repeat(rotate_matrix_y, 1, 5).t();

                cv::Mat p_rotate_x;
                cv::reduce(rotate_matrix_x.mul(p), p_rotate_x, 1, cv::REDUCE_SUM, CV_32F);

                cv::Mat p_rotate_y;
                cv::reduce(rotate_matrix_y.mul(p), p_rotate_y, 1, cv::REDUCE_SUM, CV_32F);

                cv::Mat p_rotate;
                cv::hconcat( p_rotate_x, p_rotate_y, p_rotate);

                cv::Mat p3_in_origin = origin - p_rotate(cv::Range(4, 5), cv::Range(0, 2));

                vector<cv::Point2f> roi;
                for (int k = 0; k < 4; k++)
                {
                    cv::Mat result = p_rotate(cv::Range(k, k+1), cv::Range(0, 2)) + p3_in_origin;
                    roi.push_back(cv::Point2f(result.at<float>(0, 0), result.at<float>(0, 1)));
                }
                bboxes.push_back(cv::RotatedRect(roi[0], roi[1], roi[2]));  
                rois.push_back(roi);
            }
        }
    }
}
