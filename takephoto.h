#pragma once

#include <QDebug>
#include <QDir>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class StereoPhotoTaker
{
public:
    StereoPhotoTaker(cv::VideoCapture* cap = nullptr);
    int  takePhoto();
    bool deletePhotos();
    void setVideoCapture(cv::VideoCapture* cap);

    QString getImageDirRelPath() const;
    QString getLeftImageNamePattern() const;
    QString getRightImageNamePattern() const;

private:
    int           imageIndex_;
    const QString fileExtension_;
    const QString leftImageNamePrifx_;
    const QString rightImageNamePrifx_;
    const QString imageDirRelPath_;

    cv::VideoCapture* cap_;
};
