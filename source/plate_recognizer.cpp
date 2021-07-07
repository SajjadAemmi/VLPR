#include <iostream>

#include "plate_recognizer.h"   

using namespace std;


PlateRecognizer::PlateRecognizer() 
{
    net = cv::dnn::readNet("models/plate_recognizer.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    mean = cv::Scalar(123.68, 116.78, 103.94);
    plate_width = 320;
    plate_height = 64;
    scalefactor = 1.0;
    score_threshold = 0.90;
    alphabet = "##0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";
}

PlateRecognizer::~PlateRecognizer()
{
    
}

bool PlateRecognizer::recognize(Plate& plate)
{
    cv::Mat test;
    start_time = clock();
    
    cv::dnn::blobFromImage(plate.image, blob, scalefactor, cv::Size(plate_width, plate_height), mean, true, false); // Create a 4D blob from a plate image.
    
    net.setInput(blob);
    net.forward(outs);

    cv::Mat input = outs.reshape(1, vector<int> {outs.size[0], outs.size[2]});
    plate.text = postProcess(input);

    end_time = clock();
    cout << "recognize forward time: " << float(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;

    return plate.text->size() > 0;
}

string PlateRecognizer::postProcess(cv::Mat input)
{
    cv::Mat softmax_result, values, probs, final_probs;
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::Scalar mean;
    string text = "";

    softmax(input, softmax_result);
    for (int i = 0; i < softmax_result.size[0]; i++)
    {
        cv::minMaxLoc(softmax_result.row(i), &minVal, &maxVal, &minLoc, &maxLoc);
        probs.push_back(maxVal);
        values.push_back(maxLoc.x);
    }
//    cout << "softmax_result" << softmax_result << endl;

    for (int i = 0; i < values.size[0]; i++)
    {
        cout << values.at<int>(i) << endl;
        if(values.at<int>(i) != 0 && !(i > 0 && values.at<int>(i) == values.at<int>(i-1)))
        {
            text += alphabet[values.at<int>(i)];
            final_probs.push_back(probs.at<double>(i));
        }
    }
    mean = cv::mean(final_probs);
    cout << "score " << mean[0] << endl;

    if(mean[0] > score_threshold)
        return text;
    else
        return "";
}

void softmax(cv::Mat input, cv::Mat& output)
{
    cv::Mat e_x;
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    for (int i = 0; i < input.size[0]; i++)
    {
        cv::minMaxLoc(input.row(i), &minVal, &maxVal, &minLoc, &maxLoc);
        cv::exp(input.row(i) - maxVal, e_x);
        output.push_back(e_x / cv::sum(e_x));
    }
}
