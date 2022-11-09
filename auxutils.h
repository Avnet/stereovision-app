#pragma once

#include <QImage>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/imgproc/types_c.h>

class AuxUtils
{
public:
    AuxUtils();
    ~AuxUtils();
    AuxUtils(AuxUtils const&) = delete;
    AuxUtils operator=(AuxUtils const&) = delete;

    // CONVERTERS
    static cv::Mat convertQImageToMat(const QImage& src);
    static QImage  convertMatToQImage(const cv::Mat& src);
};
