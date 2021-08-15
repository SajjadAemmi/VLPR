#include <iostream>
#include "QFile"
#include "QDir"
#include "QtDebug"
#include "QMessageBox"
#include "QFileDialog"
#include "QTextStream"
#include "mainwindow.h"
#include "time.h"

using namespace std;


MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->btn_browse, SIGNAL(clicked()), this, SLOT(browse()));
    connect(ui->btn_play, SIGNAL(clicked()), this, SLOT(play()));
    connect(ui->btn_pause, SIGNAL(clicked()), this, SLOT(pause()));
    connect(ui->btn_stop, SIGNAL(clicked()), this, SLOT(stop()));

    process = new Process();
    connect(process, SIGNAL(signalShowPreview(cv::Mat)), this, SLOT(slotShowPreview(cv::Mat)));
    connect(process, SIGNAL(signalShowPlate(cv::Mat, QString)), this, SLOT(slotShowPlate(cv::Mat, QString)));
    connect(process, SIGNAL(signalUpdateClassCounters(int)), this, SLOT(slotUpdateClassCounters(int)));

    connect(ui->slider, SIGNAL(valueChanged(int)), process, SLOT(slotChangeSlidervalue(int)));

    for (int i = 0; i < 5; i++)
        this->lbl_class_ids[i] = MainWindow::findChild<QLabel *>("lbl_class_" + QString::number(i));

    ui->tableWidget->setColumnCount(2);
    ui->tableWidget->setRowCount(1);
}


MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::browse()
{
    image_path = QFileDialog::getOpenFileName(this, tr("Open Image File"), "/home", tr("MP4 (*.mp4 *.MP4)"));
    ui->tb_path->setText(image_path);

    video_path_str = image_path.toUtf8().constData();

    cap = cv::VideoCapture(video_path_str);

    cv::Mat frame;
    cap >> frame;
    slotShowPreview(frame);
}


void MainWindow::slotShowPlate(cv::Mat image, QString text)
{
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    QImage qimage = QImage((uchar*) image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(qimage);

    QTableWidgetItem *imageItem = new QTableWidgetItem();
    imageItem->setData(Qt::DecorationRole, pixmap);
    ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, 1, imageItem);

    QTableWidgetItem *textItem = new QTableWidgetItem();
    textItem->setText(text);
    ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, 0, textItem);

    ui->tableWidget->insertRow(ui->tableWidget->rowCount());
}


void MainWindow::slotShowPreview(cv::Mat image)
{
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    QImage qimage = QImage((uchar*) image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    QPixmap pixmap = QPixmap::fromImage(qimage);
    ui->lbl_preview->setPixmap(pixmap);
}

void MainWindow::slotUpdateClassCounters(int class_id)
{
    if(class_id < 5)
    {
        int count = this->lbl_class_ids[class_id]->text().toInt();
        count++;
        this->lbl_class_ids[class_id]->setText(QString::number(count));
    }
}


void MainWindow::pause()
{

}

void MainWindow::stop()
{

}

void MainWindow::play()
{
    process->video_path_str = video_path_str;
    process->start();
}
