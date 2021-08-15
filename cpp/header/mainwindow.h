#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>

#include <QMainWindow>
#include "QLineEdit"
#include "ui_mainwindow.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "process.h"

using namespace std;


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    clock_t start_time;
    clock_t end_time;
    cv::dnn::Net net;
    cv::Mat image;
    QString image_path;
    string video_path_str;
    cv::VideoCapture cap;
    Process *process;

    QLabel *lbl_class_ids[5];

public slots:
    void browse();
    void play();
    void pause();
    void stop();
    void slotShowPreview(cv::Mat);
    void slotShowPlate(cv::Mat, QString);
    void slotUpdateClassCounters(int class_id);

};
