#include "mainwindow.h"
#include <algorithm>
#include <iostream>
#include <list>
#include <QGLWidget>
#include <QTimer>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "ui_mainwindow.h"

using namespace cv;
using namespace std;

#ifdef ARCH_PC
const QString MainWindow ::tfliteFileLong_  = APP_DATA_DIR "/dnnData/face_detection_full_range.tflite";
const QString MainWindow ::tfliteFileShort_ = APP_DATA_DIR "/dnnData/face_detection_short_range.tflite";
#else
const QString MainWindow ::tfliteFileLong_  = APP_DATA_DIR "/dnnData/face_detection_full_range.tflite";
const QString MainWindow ::tfliteFileShort_ = APP_DATA_DIR "/dnnData/face_detection_short_range_int8.tflite";
#endif

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui_(new Ui::MainWindow)
    , stereoPhotoTaker_(make_unique<StereoPhotoTaker>(nullptr))
    , calibrationThread_(nullptr)
    , faceDetectionShortThread_(nullptr)
    , faceDetectionLongThread_(nullptr)
    , faceDetectorShort_(false)
    , faceDetectorLong_(true)
{
    ui_->setupUi(this);
    qDebug() << qVersion();

    auto glleft = new QGLWidget(this);
    ui_->graphicsViewLeft->setViewport(glleft);
    ui_->graphicsViewLeft->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
    ui_->graphicsViewLeft->setScene(new QGraphicsScene(this));
    ui_->graphicsViewLeft->scene()->addItem(&pixmapLeft_);

    auto glright = new QGLWidget(this);
    ui_->graphicsViewRight->setViewport(glright);
    ui_->graphicsViewRight->setViewportUpdateMode(QGraphicsView::NoViewportUpdate);
    ui_->graphicsViewRight->setScene(new QGraphicsScene(this));
    ui_->graphicsViewRight->scene()->addItem(&pixmapRight_);

    connect(this, &MainWindow::imageToShow, this, &MainWindow::showImage);
    connect(this, &MainWindow::imageToShow, this, &MainWindow::updateFPMInfo);

    calibrationThread_ = new QThread(this);

    faceDetectionShortThread_ = new QThread(this);
    faceDetectorShort_.moveToThread(faceDetectionShortThread_);
    connect(faceDetectionShortThread_, &QThread::started, &faceDetectorShort_, &FaceDetection::handleRequist);
    connect(&faceDetectorShort_, &FaceDetection::result, this, &MainWindow::handleFaceDetectionResult);

    faceDetectionLongThread_ = new QThread(this);
    faceDetectorLong_.moveToThread(faceDetectionLongThread_);
    connect(faceDetectionLongThread_, &QThread::started, &faceDetectorLong_, &FaceDetection::handleRequist);
    connect(&faceDetectorLong_, &FaceDetection::result, this, &MainWindow::handleFaceDetectionResult);

#ifdef ARCH_PC
    // set default path of camera
    ui_->videoEdit->setText("compositor name=comp sink_1::xpos=640 sink_1::ypos=0 ! videoconvert ! appsink v4l2src device=/dev/video2 ! video/x-raw,format=YUY2,width=640,height=480 ! videoconvert ! video/x-raw,format=RGB ! comp. v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480 ! videoconvert ! video/x-raw,format=RGB ! comp.");
#else
    ui_->videoEdit->setText("v4l2src device=/dev/video2 ! video/x-raw,width=1280,height=400 ! appsink");
#endif

    TESTVarSetDefault();
    TESTVarUpdate();
}

MainWindow::~MainWindow()
{
    delete ui_;
}

void MainWindow::TESTVarSetDefault()
{
    ui_->TESTValue1->setValue(0.6);  // Face detection confidence threshold
    ui_->TESTValue2->setValue(0.75); // Face detection overlap threshold, 0.75 means if two face blocks in a frame has a more than 75% overlapped area, then the second one will be filtered.
    ui_->TESTValue3->setValue(0.2);  // width reletive deviation threshold, 0.1 means that the face blocks in left and right frame will not be matched each other if the difference of width more than 10%.
    ui_->TESTValue4->setValue(30);   // max of deviation of disparity v. The valid pair of feature points, whose difference of v-coordinats should not bigger than this value
    ui_->TESTValue5->setValue(0.2);  // height reletive deviation threshold, 0.1 means that the face blocks in left and right frame will not be matched each other if the difference of height more than 10%.
    ui_->TESTValue6->setValue(1);    // // for the acceleration, we do not process all frame but process one frame every skipFrame frames. 4 means that in every 4 frame will be 1 frame processed for the acceleration.

    ui_->Face->setChecked(false);          // enalbe face detection or not
    ui_->Rectification->setChecked(false); // enalbe Rectification or not

    ui_->deletePhotos->setStyleSheet("background-color: red");
}
void MainWindow::TESTVarUpdate()
{
    TESTVar1 = ui_->TESTValue1->value();
    TESTVar2 = ui_->TESTValue2->value();
    TESTVar3 = ui_->TESTValue3->value();
    TESTVar4 = ui_->TESTValue4->text().toInt();
    TESTVar5 = ui_->TESTValue5->value();
    TESTVar6 = ui_->TESTValue6->text().toInt();

    TESTCheckBoxFace = ui_->Face->isChecked();
    TESTCheckBoxRect = ui_->Rectification->isChecked();

    faceDetectorShort_.setMinScoreThresh(TESTVar1);
    faceDetectorShort_.setOverlapThresh(TESTVar2);

    faceDetectorLong_.setMinScoreThresh(TESTVar1);
    faceDetectorLong_.setOverlapThresh(TESTVar2);

    unitMeter_     = ui_->radioButton_meter->isChecked();
    faceModelLong_ = ui_->radioButton_long->isChecked();
}

bool MainWindow::openCamera(const QString& videoPath, VideoCapture& videoCap)
{
    auto isCamera    = false;
    auto cameraIndex = videoPath.toInt(&isCamera);
    if (isCamera) {
        if (!videoCap.open(cameraIndex)) {
            QString msg = QString("Make sure you entered a correct camera index: ") + videoPath;
            msgBox("Camera Error", msg, QMessageBox::Critical);
            return false;
        }
    } else {
        if (!videoCap.open(videoPath.trimmed().toStdString())) {
            QString msg = QString("Make sure you entered a correct and supported video file path: ") + videoPath;
            msgBox("Video Error", msg, QMessageBox::Critical);
            return false;
        }
    }
    return true;
}

void MainWindow::updateFPMInfo()
{
    timer_.update();
    ui_->TESTFPS->setValue(timer_.FPM());
    ui_->TESTTimeElapsedProFrame->setValue(timer_.avgTimePF());
}

void MainWindow::handleCalibrationResult(int identifier)
{
    ui_->Calibrate->setEnabled(true);

    calibrationThread_->quit();
    calibrationThread_->wait();
    if (identifier == 0) {
        msgBox("Information", "Calibration complete!", QMessageBox::Information, 700);
        loadCalibrationData();
    } else if (identifier == 1) {
        msgBox("Calibration Error", "Find key points in chess board but can not pair them in stereo images. Failed to calibrate stereo camera!", QMessageBox::Warning);
    } else {
        msgBox("Calibration Error", "Can not find key points in chess board. Failed to calibrate camera!", QMessageBox::Warning);
    }
}

void MainWindow::msgBox(const QString& titel, const QString& content, QMessageBox::Icon icon, int msec)
{
    QMessageBox* msgBox = new QMessageBox(this);
    msgBox->setAttribute(Qt::WA_DeleteOnClose); //makes sure the msgbox is deleted automatically when closed
    msgBox->setIcon(icon);
    msgBox->setStandardButtons(QMessageBox::Ok);
    msgBox->setWindowTitle(titel);
    msgBox->setText(content);
    msgBox->setModal(false);

    if (msec != 0) // it is a auto closed msgBox
    {
        msgBox->setStandardButtons(QMessageBox::NoButton);
        QTimer* timer = new QTimer(this);
        connect(timer, &QTimer::timeout, msgBox, &QDialog::accept);
        timer->start(msec);
    }

    auto* desktop = QApplication::desktop();
    auto  x       = (desktop->width() - msgBox->width()) / 2;
    auto  y       = (desktop->height() - msgBox->height()) / 2;
    msgBox->move(x, y);
    msgBox->show();
}

void MainWindow::loadCalibrationData(const QString& filePath)
{
    if (calibrationDataLoaded_)
        return;
    FileStorage stereoData(filePath.toStdString(), FileStorage::READ);
    if (!stereoData.isOpened()) {
        msgBox("Calibration Error", "Can not open the camera data file! Please calibrate cameras first!", QMessageBox::Critical);
        return;
    }
    stereoData["KL"] >> st_.KL;
    stereoData["KR"] >> st_.KR;
    stereoData["DL"] >> st_.DL;
    stereoData["DR"] >> st_.DR;
    stereoData["R"] >> st_.R;
    stereoData["T"] >> st_.T;
    stereoData["RL"] >> st_.RL;
    stereoData["RR"] >> st_.RR;
    stereoData["PL"] >> st_.PL;
    stereoData["PR"] >> st_.PR;
    stereoData["Q"] >> st_.Q;

    calibrationDataLoaded_ = true;
}

void MainWindow::rectifyImage(const Mat& frameInputL, const Mat& frameInputR, Mat& frameOutputL, Mat& frameOutputR)
{
    if (!calibrationDataLoaded_)
        loadCalibrationData();
    if (!calibrationDataLoaded_) {
        ui_->Face->setChecked(false);
        ui_->Rectification->setChecked(false);
        TESTVarUpdate();
        frameOutputL = frameInputL;
        frameOutputR = frameInputR;
        return;
    }
    Mat mapxL, mapyL, mapxR, mapyR;

    initUndistortRectifyMap(st_.KR, st_.DR, st_.RR, st_.PR, frameInputR.size(), CV_32F, mapxR, mapyR);
    initUndistortRectifyMap(st_.KL, st_.DL, st_.RL, st_.PL, frameInputL.size(), CV_32F, mapxL, mapyL);

    remap(frameInputL, frameOutputL, mapxL, mapyL, INTER_LINEAR);
    remap(frameInputR, frameOutputR, mapxR, mapyR, INTER_LINEAR);
}

double MainWindow::calDistance(const cv::Rect& objL, const cv::Rect& objR, const cv::Mat& Q)
{
    if (!calibrationDataLoaded_) {
        QString msg = QString("Can not find calibration data!");
        QMessageBox::critical(this, "Calibration Error", msg);
        ui_->Face->setChecked(false);
        ui_->Rectification->setChecked(false);
        TESTVarUpdate();
        return 0;
    }
    auto xL = objL.x + objL.width / 2.0;
    auto yL = objL.y + objL.height / 2.0;
    auto xR = objR.x + objR.width / 2.0;
    auto yR = objR.y + objR.height / 2.0;

    double focal    = Q.at<double>(2, 3);
    double cx       = abs(Q.at<double>(0, 3));
    double cy       = abs(Q.at<double>(1, 3));
    double baseline = abs(1. / Q.at<double>(3, 2));
    double z        = focal * baseline / (abs(xL - xR) * 1000.0); // convert to meter
    double x        = (xL - cx) * z / focal;
    double y        = (yL - cy) * z / focal;
    double dis      = sqrt(x * x + y * y + z * z);

    if (debugMode_) {
        std::cout << "Image coordinate Left x: " << xL << " Left y: " << yL << " Right x: " << xR << " Right y: " << yR << std::endl;
        std::cout << "disparity y: " << yL - yR << std::endl;
        std::cout << "disparity x: " << xL - xR << std::endl;
        std::cout << "World coordinate x: " << x << " y: " << y << " z: " << z << " distance: " << dis << " m." << std::endl;
    }

    return dis; // return distance in meter
}

void MainWindow::handleFaceDetectionResult(cv::Mat frameLeft, cv::Mat frameRight, std::vector<Rect> objLeft, std::vector<Rect> objRight)
{
    if (debugMode_) {
        std::cout << "objsL.size(): " << objLeft.size() << std::endl;
        std::cout << "objsR.size(): " << objRight.size() << std::endl;
    }

    for (auto obj : objLeft)
        rectangle(frameLeft, obj, CV_RGB(0, 255, 0), 2);
    for (auto obj : objRight)
        rectangle(frameRight, obj, CV_RGB(0, 255, 0), 2);

    auto ret = matchObjs(objLeft, objRight);
    for (auto obj : ret) {
        auto dis = calDistance(obj.first, obj.second, st_.Q);
        if (dis < 0.1)
            break;

        string showText;
        if (!unitMeter_) {
            constexpr double meter2ft = 3.28084;
            dis                       = dis * meter2ft;
            int ft                    = static_cast<int>(dis);
            int inch                  = static_cast<int>((dis - static_cast<double>(ft)) * 12);
            showText                  = std::to_string(ft) + string("ft") + std::to_string(inch) + string("in");
        } else {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(1) << dis;
            showText = stream.str() + string("m");
        }

        Point textOrigin(obj.first.x, obj.first.y + obj.first.height);
        putText(frameLeft, showText, textOrigin, FONT_HERSHEY_PLAIN, 3, CV_RGB(0, 255, 0), 3, 8);
    }

    emit imageToShow(frameLeft, frameRight);
}

std::vector<std::pair<Rect, Rect>> MainWindow::matchObjs(const std::vector<Rect>& inputLeft, const std::vector<Rect>& inputRight)
{
    std::vector<std::pair<Rect, Rect>> ret;
    list<Rect>                         inputLeftList(inputLeft.cbegin(), inputLeft.cend());
    list<Rect>                         inputRightList(inputRight.cbegin(), inputRight.cend());

    auto comp = [](const Rect& a, const Rect& b) { return a.x > b.x; }; // left first
    while (!inputLeftList.empty()) {
        std::priority_queue<Rect, std::vector<Rect>, decltype(comp)> queueLeft(comp);
        std::priority_queue<Rect, std::vector<Rect>, decltype(comp)> queueRight(comp);

        auto it  = inputLeftList.begin();
        auto ref = *it;
        while (it != inputLeftList.end()) {
            auto widthRel   = static_cast<double>(it->width) / static_cast<double>(ref.width);
            auto heighthRel = static_cast<double>(it->height) / static_cast<double>(ref.height);
            auto yDiff      = it->y - ref.y;
            bool valid      = (abs(widthRel - 1) < TESTVar3) && (abs(heighthRel - 1) < TESTVar5) && (abs(yDiff) < TESTVar4);
            if (valid) {
                queueLeft.push(std::move(*it));
                it = inputLeftList.erase(it);
            } else
                it++;
        }

        it = inputRightList.begin();
        while (it != inputRightList.end()) {
            auto widthRel   = static_cast<double>(it->width) / static_cast<double>(ref.width);
            auto heighthRel = static_cast<double>(it->height) / static_cast<double>(ref.height);
            auto yDiff      = it->y - ref.y;
            bool valid      = (abs(widthRel - 1) < TESTVar3) && (abs(heighthRel - 1) < TESTVar5) && (abs(yDiff) < TESTVar4);
            if (valid) {
                queueRight.push(std::move(*it));
                it = inputRightList.erase(it);
            } else
                it++;
        }

        auto size = queueLeft.size() > queueRight.size() ? queueRight.size() : queueLeft.size();
        for (auto i = 0; i < size; i++) {
            ret.emplace_back(queueLeft.top(), queueRight.top());
            queueLeft.pop();
            queueRight.pop();
        }
    }
    return ret;
}

void MainWindow::showImage(const cv::Mat& frameLeftInput, const cv::Mat& frameRightInput)
{
    Mat frameRgbL, frameRgbR;
    cvtColor(frameLeftInput, frameRgbL, CV_BGR2RGB);
    cvtColor(frameRightInput, frameRgbR, CV_BGR2RGB);

    QImage::Format       format      = QImage::Format_RGB888;
    QGraphicsPixmapItem* pixmapPtr   = &pixmapRight_;
    QGraphicsView*       graphicView = ui_->graphicsViewRight;
    QImage               rightImg(frameRgbR.data, frameRgbR.cols, frameRgbR.rows, frameRgbR.step, format);
    pixmapPtr->setPixmap(QPixmap::fromImage(rightImg));
    graphicView->fitInView(pixmapPtr, Qt::KeepAspectRatio);

    pixmapPtr   = &pixmapLeft_;
    graphicView = ui_->graphicsViewLeft;
    QImage leftImg(frameRgbL.data, frameRgbL.cols, frameRgbL.rows, frameRgbL.step, format);
    pixmapPtr->setPixmap(QPixmap::fromImage(leftImg));
    graphicView->fitInView(pixmapPtr, Qt::KeepAspectRatio);
}

void MainWindow::onDeletePhotosPressed()
{
    if (!stereoPhotoTaker_->deletePhotos()) {
        msgBox("Error", "Failed to delete photos!", QMessageBox::Critical);
    } else {
        msgBox("Information", "Photos are deleted!", QMessageBox::Information, 700);
    }
}

void MainWindow::onTakephotoPressed()
{
    auto index = stereoPhotoTaker_->takePhoto();
    if (index == 0) {
        msgBox("Error", "Failed to take photos! Please start application and check video stream", QMessageBox::Critical);
        return;
    }
    msgBox("Information", "Saving image pair " + QString::number(index), QMessageBox::Information, 400);
}

void MainWindow::onCalibratePressed()
{
    ui_->Calibrate->setEnabled(false);

    StereoCalibrator* calibrator = new StereoCalibrator(stereoPhotoTaker_->getLeftImageNamePattern(), stereoPhotoTaker_->getRightImageNamePattern(), stereoPhotoTaker_->getImageDirRelPath());
    calibrator->moveToThread(calibrationThread_);
    connect(calibrationThread_, &QThread::started, calibrator, &StereoCalibrator::calibrate);
    connect(calibrator, &StereoCalibrator::calibrationResult, this, &MainWindow::handleCalibrationResult);
    connect(calibrationThread_, &QThread::finished, calibrator, &QObject::deleteLater);

    QMessageBox* msgBox = new QMessageBox(this);
    msgBox->setText("Calibration started please wait...");
    msgBox->setIcon(QMessageBox::Information);
    msgBox->setStandardButtons(QMessageBox::NoButton);
    connect(calibrationThread_, &QThread::finished, msgBox, &QDialog::accept);
    msgBox->setModal(false);
    auto* desktop = QApplication::desktop();
    auto  x       = (desktop->width() - msgBox->width()) / 2;
    auto  y       = (desktop->height() - msgBox->height()) / 2;
    msgBox->move(x, y);
    msgBox->show();

    calibrationThread_->start();
}

void MainWindow::onStartBtnPressed()
{
    // open camera
    if (video_.isOpened()) {
        ui_->startBtn->setText("Start");
        video_.release();
        return;
    }
    auto cameraOpen = openCamera(ui_->videoEdit->text(), video_);
    if (!cameraOpen) {
        video_.release();
        return;
    }
    ui_->startBtn->setText("Stop");
    auto skip = 0;

    stereoPhotoTaker_->setVideoCapture(&video_);

    timer_.start();
    while (video_.isOpened()) {
        Mat original, frameL, frameR;
        video_ >> original;

        // divide video stream to two
        auto width = original.cols;
        frameR     = original(Range::all(), Range(0, width / 2));
        frameL     = original(Range::all(), Range(width / 2, width));

        if (frameL.empty() || frameR.empty()) {
            msgBox("Video Error", "Cannot read data from opened camera!", QMessageBox::Critical);
            return;
        }

        skip++;
        if (skip != TESTVar6) {
            continue;
        } else {
            skip = 0;
        }

        // initialize tensorflow at the first time
        if (!faceDetectorInit_) {
            std::cout << "init tensor" << std::endl;
            faceDetectorShort_.initialize(frameL.rows, frameL.cols, tfliteFileShort_);
            faceDetectorLong_.initialize(frameL.rows, frameL.cols, tfliteFileLong_);
            faceDetectorInit_ = true;
        }

        // image rectification
        if (TESTCheckBoxFace || TESTCheckBoxRect) {
            rectifyImage(frameL, frameR, frameL, frameR);
        }

        // detection
        if (TESTCheckBoxFace) {
            if (!TESTCheckBoxRect) {
                ui_->Rectification->setChecked(true);
                TESTVarUpdate();
            }

            if (faceModelLong_) {
                faceDetectorLong_.setFrameL(frameL);
                faceDetectorLong_.setFrameR(frameR);
                if (!faceDetectionLongThread_->isRunning()) {
                    faceDetectionLongThread_->start();
                }
            } else {
                faceDetectorShort_.setFrameL(frameL);
                faceDetectorShort_.setFrameR(frameR);

                if (!faceDetectionShortThread_->isRunning()) {
                    faceDetectionShortThread_->start();
                }
            }
        } else {
            emit imageToShow(frameL, frameR);
        }

        qApp->processEvents();
    }

    ui_->startBtn->setText("Start");
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    if (video_.isOpened()) {
        ui_->startBtn->setText("Start");
        video_.release();
    }

    if (!calibrationThread_->wait(100)) {
        msgBox("Warning", "Waiting for calibration finished", QMessageBox::Warning);
    }
    calibrationThread_->quit();
    calibrationThread_->wait();

    if (!faceDetectionShortThread_->wait(100)) {
        msgBox("Warning", "Waiting for tensorflow stop", QMessageBox::Warning);
    }
    faceDetectionShortThread_->quit();
    faceDetectionShortThread_->wait();

    if (!faceDetectionLongThread_->wait(100)) {
        msgBox("Warning", "Waiting for tensorflow stop", QMessageBox::Warning);
    }
    faceDetectionLongThread_->quit();
    faceDetectionLongThread_->wait();

    event->accept();
}
