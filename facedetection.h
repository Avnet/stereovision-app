#pragma once

#include <optional>
#include <vector>
#include "auxutils.h"
#include "opencv2/calib3d.hpp"
#include "tensorflow.h"

typedef std::vector<Point2d> FaceRealignmentMatrix;

struct FaceOptions
{
    int numValuesPerKeypoint = 2;
    int numKeyPoints         = 0;
    int numClasses;
    int numBoxes;
    int numCoords;

    int        boxCoordOffset = 0;
    int        keyPointCoordOffset;
    QList<int> ignoreClasses;

    double scoreClippingThresh;
    double overlapThresh;
    double minScoreThresh;

    double xScale = 0.0;
    double yScale = 0.0;
    double wScale = 0.0;
    double hScale = 0.0;

    bool applyExponentialOnBoxSize = false;
    bool reverseOutputOrder        = true;
    bool flipVertically            = false;
    bool sigmoidScore              = true;
};

struct AnchorOptions
{
    int inputSizeWidth;
    int inputSizeHeight;

    double minScale;
    double maxScale;
    double anchorOffsetX;
    double anchorOffsetY;

    int        numLayers;
    QList<int> strides;
    QList<int> featureMapWidth;
    QList<int> featureMapHeight;

    QList<double> aspectRatios;
    double        interpolatedScaleAspectRatio;

    bool fixedAnchorSize;
    bool reduceBoxesInLowestLayer;

    int stridesSize() { return strides.length(); }

    int featureMapWidthSize() { return featureMapWidth.length(); }

    int featureMapHeightSize() { return featureMapHeight.length(); }
};

struct Anchor
{
    double xCenter;
    double yCenter;
    double h;
    double w;
};

struct Detection
{
    int    classId;
    double height;
    double width;
    double score;
    double xMin;
    double yMin;
    double xLeftEye;
    double yLeftEye;
    double xRightEye;
    double yRightEye;
    double xNose;
    double yNose;
    double xMouth;
    double yMouth;
    double xLeftEar;
    double yLeftEar;
    double xRightEar;
    double yRightEar;
};

class FaceDetection final : public TensorFlow
{
    Q_OBJECT
signals:
    void result(cv::Mat frameLeft, cv::Mat frameRight, std::vector<Rect> objLeft, std::vector<Rect> objRight);

public slots:
    void handleRequist();

public:
    FaceDetection(bool longRange);
    ~FaceDetection();
    FaceDetection(FaceDetection const&) = delete;
    FaceDetection operator=(FaceDetection const&) = delete;

    using TensorFlow::initialize;
    virtual bool initialize(int imageHeight, int imageWidth, QString modelPath);

    std::optional<std::vector<Rect>> detect(const Mat& mat, bool skip = false);

    QList<int>       getTensorShape(TfLiteTensor* tensor);
    bool             extractTensorData(QList<double>& outClass, QList<double>& outRegre);
    QList<Anchor>    extractSSDAnchors(AnchorOptions options);
    QList<Detection> extractDetections(FaceOptions   options,
                                       QList<double> rawScores,
                                       QList<double> rawBoxes,
                                       QList<Anchor> anchors);

    QList<Detection> convertPropositionsToDetections(QList<double> boxes,
                                                     QList<double> scores,
                                                     QList<int>    classes,
                                                     FaceOptions   options);

    Detection convertPropositionToDetection(
        double boxYMin, double boxXMin, double boxYMax, double boxXMax,
        double xLeftEye, double yLeftEye, double xRightEye, double yRightEye,
        double xNose, double yNose, double xMouth, double yMouth, double xLeftEar,
        double yLeftEar, double xRightEar, double yRightEar, double score,
        int classId, bool flipVertically);
    QList<double>    decodeBoxes(QList<double> rawBoxes, QList<Anchor> anchors,
                                 FaceOptions options);
    QList<Detection> filterDetections(QList<Detection> detections,
                                      double           threshold);

    void setFrameL(cv::Mat frame) { frameL_ = frame; }
    void setFrameR(cv::Mat frame) { frameR_ = frame; }

    // for Debug
    void setMinScoreThresh(double val)
    {
        faceOptions_.minScoreThresh = val;
    }

    void setOverlapThresh(double val)
    {
        faceOptions_.overlapThresh = val;
    }

private:
    bool      longRange_;
    const int targetSize_;

    cv::Mat       frameL_;
    cv::Mat       frameR_;
    FaceOptions   faceOptions_;
    AnchorOptions anchorOptions_;

    QList<double> quickSort(QList<double> a);
    QList<double> sum(double a, QList<double> b);
    QList<double> divide(QList<double> a, QList<double> b);
    QList<double> subtract(QList<double> a, QList<double> b);
    QList<double> multiply(QList<double> a, QList<double> b);
    QList<double> maximum(double value, QList<double> itemIndex);
    QList<double> itemIndex(QList<double> items, QList<int> positions);
    double        calScale(double min, double max, int strideIndex, int numStrides);

    Detection scaleDetection(Detection detection, const QImage& image);
};
