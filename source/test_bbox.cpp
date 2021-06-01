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
    float d0, d1, d2, d3, w, h, angle, x, y;

    cv::Mat image = cv::imread("input/2_crop.jpg");
    int resize_w = 1280, resize_h = 960;
    cv::resize(image, image, cv::Size(1280, 960));

    cv::dnn::Net net = cv::dnn::readNet("models/plate_detector.pb");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    vector<cv::Mat> outs;
    cv::Mat blob;
    vector<string> outNames = {"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"};
    cv::Scalar mean = cv::Scalar(123.68, 116.78, 103.94);
    float scalefactor = 1.0;

    cv::dnn::blobFromImage(image, blob, scalefactor, cv::Size(resize_w, resize_h), mean, true, false); // Create a 4D blob from a frame.
    net.setInput(blob);
    net.forward(outs, outNames);
//    end_time = clock();
//    cout << "detect forward time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;

    cv::Mat scores_raw = outs[0].reshape(1, vector<int> {1, outs[0].size[2], outs[0].size[3]});
    cv::Mat bboxes_raw = outs[1].reshape(1, vector<int> {5, outs[1].size[2], outs[1].size[3]});
 
    float i = 155, j = 120;
    d0 = bboxes_raw.at<float>(0, i, j);
    d1 = bboxes_raw.at<float>(1, i, j);
    d2 = bboxes_raw.at<float>(2, i, j);
    d3 = bboxes_raw.at<float>(3, i, j);
    angle = bboxes_raw.at<float>(4, i, j);

    cout << d0 << "\t" << d1 << "\t" << d2 << "\t" << d3 << "\t" << angle << endl;
    cout << "score " << scores_raw.at<float>(0, i, j);

    float origin_array[2] = {j*4, i*4};
    cv::Mat origin = cv::Mat(1, 2, CV_32F, origin_array);
    cout << "origin" << origin << endl;

    float data[10] = {0, -d0 - d2, d1 + d3, -d0 - d2, d1 + d3, 0, 0, 0, d3, -d2};
    cv::Mat p = cv::Mat(5, 2, CV_32F, data);
    cout << "p" << p << endl;

    float rotate_matrix_x_array[2] = {cos(angle), sin(angle)};
    cv::Mat rotate_matrix_x = cv::Mat(2, 1, CV_32F, rotate_matrix_x_array);
    rotate_matrix_x = cv::repeat(rotate_matrix_x, 1, 5).t();

    cout << "rotate_matrix_x" << rotate_matrix_x << endl;

    float rotate_matrix_y_array[2] = {-sin(angle), cos(angle)};
    cv::Mat rotate_matrix_y = cv::Mat(2, 1, CV_32F, rotate_matrix_y_array);
    rotate_matrix_y = cv::repeat(rotate_matrix_y, 1, 5).t();

    cout << "rotate_matrix_y" << rotate_matrix_y << endl;

    cv::Mat p_rotate_x;
    // cout << "rotate_matrix_x " << rotate_matrix_x.size << endl;
    // cout << "p " << p.size << endl;

    cv::reduce(rotate_matrix_x.mul(p), p_rotate_x, 1, cv::REDUCE_SUM, CV_32F);
    cout << "p_rotate_x" << p_rotate_x << endl;

    cv::Mat p_rotate_y;
    cv::reduce(rotate_matrix_y.mul(p), p_rotate_y, 1, cv::REDUCE_SUM, CV_32F);
    cout << "p_rotate_y" << p_rotate_y << endl;

    cv::Mat p_rotate;
    cv::hconcat( p_rotate_x, p_rotate_y, p_rotate);
    cout << "p_rotate" << p_rotate << endl;
    // cout << p_rotate.size << endl;

    cv::Mat p3_in_origin = origin - p_rotate(cv::Range(4,5), cv::Range(0,2));
    cout << "p3_in_origin" << p3_in_origin << endl;
    // new_p = np.array(, dtype=int)

    // cout << p_rotate << endl;

    cv::Point2f vertices[4];
    for (int k = 0; k < 4; k++)
    {
        cv::Mat result = p_rotate(cv::Range(k,k+1), cv::Range(0,2)) + p3_in_origin;
        
        vertices[k] = cv::Point2f(result.at<float>(0, 0), result.at<float>(0, 1));
        cout << vertices[k] << endl;
    }

    cv::Size outputSize = cv::Size(320, 64);
cv::Point2f dst_corners[4] = {                                     cv::Point2f(0, 0),
                                     cv::Point2f(outputSize.width - 1, 0),
                                     cv::Point2f(outputSize.width - 1, outputSize.height - 1),
                                     cv::Point2f(0, outputSize.height - 1)};

    cv::Mat pelak;
    cv::Mat rotationMatrix = cv::getPerspectiveTransform(vertices, dst_corners);
    cv::warpPerspective(image, pelak, rotationMatrix, outputSize);
    cv::imwrite("output/pelak.jpg", pelak);

}