#include "takephoto.h"

using namespace std;
using namespace cv;

StereoPhotoTaker::StereoPhotoTaker(cv::VideoCapture* cap)
    : imageIndex_(0)
    , fileExtension_(".jpg")
    , leftImageNamePrifx_("left")
    , rightImageNamePrifx_("right")
    , imageDirRelPath_(APP_DATA_DIR "/image")
    , cap_(cap)
{
}

QString StereoPhotoTaker::getImageDirRelPath() const
{
    return imageDirRelPath_;
}

QString StereoPhotoTaker::getLeftImageNamePattern() const
{
    return leftImageNamePrifx_ + QString("*") + fileExtension_;
}

QString StereoPhotoTaker::getRightImageNamePattern() const
{
    return rightImageNamePrifx_ + QString("*") + fileExtension_;
}

bool StereoPhotoTaker::deletePhotos()
{
    if (QDir(imageDirRelPath_).exists()) {
        QDir directory(imageDirRelPath_);
        if (directory.removeRecursively()) {
            qInfo() << "Photo deleted";
            imageIndex_ = 0;
        } else {
            qCritical() << "Failed to delete photo";
            return false;
        }
    }
    return true;
}

void StereoPhotoTaker::setVideoCapture(cv::VideoCapture* cap)
{
    cap_ = cap;
}

int StereoPhotoTaker::takePhoto()
{
    if (!cap_) {
        qCritical() << "No VideoCapture instance!";
        return 0;
    }
    if (!cap_->isOpened()) {
        qCritical() << "Can not open camera!";
        return 0;
    }

    if (!QDir(imageDirRelPath_).exists()) {
        QDir().mkdir(imageDirRelPath_);
    }
    QDir       directory(imageDirRelPath_);
    const auto dirAbsolutePath = directory.absolutePath();

    Mat original;
    *cap_ >> original;
    auto width      = original.cols;
    auto rightImage = original(Range::all(), Range(0, width / 2));
    auto leftImage  = original(Range::all(), Range(width / 2, width));

    if (leftImage.empty() || rightImage.empty()) {
        qCritical() << "Camera is opened but can not read data!";
        return 0;
    }

    imageIndex_++;
    auto leftImageFile  = dirAbsolutePath + QDir::separator() + leftImageNamePrifx_ + QString::number(imageIndex_) + fileExtension_;
    auto rightImageFile = dirAbsolutePath + QDir::separator() + rightImageNamePrifx_ + QString::number(imageIndex_) + fileExtension_;
    qInfo() << "Saving image pair " << imageIndex_;
    imwrite(leftImageFile.toStdString(), leftImage);
    imwrite(rightImageFile.toStdString(), rightImage);

    return imageIndex_;
}
