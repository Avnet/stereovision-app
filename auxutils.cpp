#include "auxutils.h"

using namespace cv;

AuxUtils::AuxUtils()
{}

AuxUtils::~AuxUtils()
{}

// CONVERTERS
QImage AuxUtils::convertMatToQImage(const Mat& src)
{
    QImage dest((const uchar*) src.data, src.cols, src.rows, src.step, QImage::Format_RGB32);

    dest.bits();

    return dest;
}

Mat AuxUtils::convertQImageToMat(const QImage& src)
{
    Mat result(src.height(), src.width(), CV_8UC4, (uchar*) src.bits(), src.bytesPerLine());

    return result;
}
