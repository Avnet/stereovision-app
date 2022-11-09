#pragma once

#include <iostream>
#include <optional>
#include <vector>
#include <QDebug>
#include <QDir>
#include <QMessageBox>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/opencv.hpp"

struct ChessBoardData
{
    int    width;
    int    height;
    double squareSize;
};

struct IntrinsicData
{
    cv::Mat K;
    cv::Mat D;
};

class StereoCalibrator : public QObject
{
    Q_OBJECT
public:
    StereoCalibrator(const QString& leftImageNamePattern, const QString& rightImageNamePattern, const QString& imageDirRelPath, const QString& cameraDataRelPath = QString(APP_DATA_DIR "/camera_data/"), const QString& leftCameraDataFileName = QString("left.yml"), const QString& rightCameraDataFileName = QString("right.yml"), const QString& stereoCameraDataFileName = QString("stereo.yml"), int chessBoardWidth = 8, int chessBoardHeigth = 6, int chessBoardSquareSize = 25);

public slots:
    void calibrate();

signals:
    void calibrationResult(int identifier);

private:
    std::optional<IntrinsicData> calibrationIntrinsic(const ChessBoardData& chessBoard, const QString& imageDir, const QString& imageNamePattern, const QString& cameraDir, const QString& cameraDataName);
    bool                         calibrationStereo(const ChessBoardData& chessBoard, const QString& imageDir, const QString& imageNamePatternL, const QString& imageNamePatternR, const QString& cameraDir, const QString& cameraDataName, const IntrinsicData& intrinsicL, const IntrinsicData& intrinsicR);

    const QString  leftImageNamePattern_;
    const QString  rightImageNamePattern_;
    const QString  imageDirRelPath_;
    const QString  cameraDataRelPath_;
    const QString  leftCameraDataFileName_;
    const QString  rightCameraDataFileName_;
    const QString  stereoCameraDataFileName_;
    ChessBoardData chessBoard_;
};
