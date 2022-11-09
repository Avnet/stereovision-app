#pragma once

#include <memory>
#include <queue>
#include <QCloseEvent>
#include <QDesktopWidget>
#include <QElapsedTimer>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QImage>
#include <QMainWindow>
#include <QMessageBox>
#include <QPixmap>
#include <QThread>

#include "calibration.h"
#include "facedetection.h"
#include "opencv2/opencv.hpp"
#include "takephoto.h"

namespace Ui {
class MainWindow;
}

class UITimer
{
public:
    UITimer() = default;
    void start()
    {
        timerGlobal_.start();
        timer_.start();
    }
    void update()
    {
        frames_++;
        auto msec = timer_.restart();
        timeQueue_.push(msec);
        timeSum_ += msec;

        if (timeQueue_.size() > 20) {
            timeSum_ -= timeQueue_.front();
            timeQueue_.pop();
        }

        if (timerGlobal_.elapsed() > 1000) {
            fpm_    = frames_;
            frames_ = 0;
            timerGlobal_.restart();
        }
    }
    int avgTimePF() const
    {
        return timeSum_ / timeQueue_.size();
    }
    int FPM() const
    {
        return fpm_;
    }

private:
    QElapsedTimer   timer_;
    QElapsedTimer   timerGlobal_;
    std::queue<int> timeQueue_;
    int             timeSum_ = 0;
    int             frames_  = 0;
    int             fpm_     = 0;
};

struct StereoData
{
    cv::Mat   DL; // distortion coefficients of left camera
    cv::Mat   DR; // distortion coefficients of right camera
    cv::Mat   KL; // intrinsic matrix of left camera
    cv::Mat   KR; // intrinsic matrix of right camera
    cv::Mat   R;  // rotation from the left to the right camera
    cv::Vec3d T;  //  translation from the left to the right camera
    cv::Mat   RL; // rectification transform for the left camera
    cv::Mat   RR; // rectification transform for the right camera
    cv::Mat   PL; // projection matrix in the new rectified coordinate system for the left camera
    cv::Mat   PR; // projection matrix in the new rectified coordinate system for the right camera
    cv::Mat   E;  // essential matrix
    cv::Mat   F;  // fundamental matrix
    cv::Mat   Q;  // disparity-to-depth mapping matrix
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = 0);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event);

signals:
    void imageToShow(const cv::Mat& frameLeft, const cv::Mat& frameRight);
    void faceDetectionRequist(cv::Mat frameLeft, cv::Mat frameRight);

private slots:
    void onStartBtnPressed();
    void onTakephotoPressed();
    void onDeletePhotosPressed();
    void onCalibratePressed();
    void TESTVarUpdate();
    void handleCalibrationResult(int identifier);
    void handleFaceDetectionResult(cv::Mat frameLeft, cv::Mat frameRight, std::vector<Rect> objLeft, std::vector<Rect> objRight);
    void showImage(const cv::Mat& frameLeftInput, const cv::Mat& frameRightInput);
    void updateFPMInfo();

private:
    void   TESTVarSetDefault();
    double TESTVar1 = 1;
    double TESTVar2 = 1;
    double TESTVar3 = 1;
    int    TESTVar4 = 1;
    double TESTVar5 = 1;
    int    TESTVar6 = 1;

    bool TESTCheckBoxFace = false;
    bool TESTCheckBoxRect = false;

private:
    bool                               openCamera(const QString& videoPath, cv::VideoCapture& videoCap);
    void                               loadCalibrationData(const QString& filePath = APP_DATA_DIR "/camera_data/stereo.yml");
    void                               rectifyImage(const cv::Mat& frameInputL, const cv::Mat& frameInputR, cv::Mat& frameOutputL, cv::Mat& frameOutputR);
    void                               objectDetection(const cv::Mat& frameInputLeft, const cv::Mat& frameInputRight, cv::Mat& frameOutputL, cv::Mat& frameOutputR);
    std::vector<std::pair<Rect, Rect>> matchObjs(const std::vector<Rect>& inputLeft, const std::vector<Rect>& inputRight);
    double                             calDistance(const cv::Rect& objL, const cv::Rect& objR, const cv::Mat& Q);

    void msgBox(const QString& titel, const QString& content, QMessageBox::Icon icon, int msec = 0);

    Ui::MainWindow* ui_;
    StereoData      st_;

    cv::VideoCapture video_;

    std::unique_ptr<StereoPhotoTaker> stereoPhotoTaker_;
    FaceDetection                     faceDetectorShort_;
    FaceDetection                     faceDetectorLong_;
    bool                              faceDetectorInit_ = false;
    QThread*                          calibrationThread_;
    QThread*                          faceDetectionShortThread_;
    QThread*                          faceDetectionLongThread_;
    UITimer                           timer_;

    QGraphicsPixmapItem pixmapLeft_;
    QGraphicsPixmapItem pixmapRight_;

    static const QString tfliteFile_;
    static const QString tfliteFileShort_;
    static const QString tfliteFileLong_;
    bool                 calibrationDataLoaded_ = false;
    bool                 debugMode_             = false;
    bool                 unitMeter_             = false;
    bool                 faceModelLong_         = false;
};
