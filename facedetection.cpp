#include "facedetection.h"
#include "math.h"
#include <QThread>

FaceDetection::FaceDetection(bool longRange)
    : longRange_(longRange)
    , targetSize_(longRange ? 192 : 128) // short range:128, back:256, full range:192
{}

FaceDetection::~FaceDetection()
{}

// WORKFLOW
bool FaceDetection::initialize(int imageHeight, int imageWidth, QString modelPath)
{
    faceOptions_ = FaceOptions{
        .numValuesPerKeypoint = 2,
        .numKeyPoints         = 6,
        .numClasses           = 1,
        .numBoxes             = longRange_ ? 2304 : 896, // short range:896, back:896, full range:2304
        .numCoords            = 16,

        .boxCoordOffset      = 0,
        .keyPointCoordOffset = 4,
        .ignoreClasses       = QList<int>(),

        .scoreClippingThresh = 100.00,
        .overlapThresh       = 0.75,
        .minScoreThresh      = settings_.detectionThreshold,

        .xScale = static_cast<double>(targetSize_),
        .yScale = static_cast<double>(targetSize_),
        .wScale = static_cast<double>(targetSize_),
        .hScale = static_cast<double>(targetSize_),

        .applyExponentialOnBoxSize = false,
        .reverseOutputOrder        = true,
        .flipVertically            = false,
        .sigmoidScore              = true};

    anchorOptions_ = AnchorOptions{
        .inputSizeWidth  = targetSize_,
        .inputSizeHeight = targetSize_,

        .minScale = 0.1484375,
        .maxScale = 0.75,

        .anchorOffsetX = 0.5,
        .anchorOffsetY = 0.5,

        .numLayers = longRange_ ? 1 : 4,                                     // short range:4, back:4, full range: 1
        .strides   = longRange_ ? QList<int>{4} : QList<int>{8, 16, 16, 16}, // short range:{8, 16, 16, 16}, back:{16, 32, 32, 32}, full range:{4}

        .featureMapWidth  = QList<int>(),
        .featureMapHeight = QList<int>(),

        .aspectRatios                 = QList<double>{1.0},
        .interpolatedScaleAspectRatio = longRange_ ? 0.0 : 1.0, // short range:1.0, back:1.0, full range:0.0

        .fixedAnchorSize          = true,
        .reduceBoxesInLowestLayer = false};

    initialized_ = TensorFlow::initialize(imageHeight, imageWidth, modelPath);
    return initialized_;
}

void FaceDetection::handleRequist()
{
    auto objLeft  = detect(frameL_);
    auto objRight = detect(frameR_);

    if (objLeft.has_value() && objRight.has_value()) {
        emit result(frameL_, frameR_, objLeft.value(), objRight.value());
        this->thread()->quit();
    } else {
        qDebug() << "Failed to detect face";
    }
}

std::optional<std::vector<Rect>> FaceDetection::detect(const Mat& mat, bool skip)
{
    if (skip) {
        std::vector<Rect> ret;
        return ret;
    }
    Mat matRGBA;
    cvtColor(mat, matRGBA, CV_BGR2RGBA); // Qimage should be RGB32, it is RGBA in opencv
    auto image = AuxUtils::convertMatToQImage(matRGBA);
    if (image.isNull()) {
        qDebug() << "Cannot run face detection pipeline due to empty image...";
        return {};
    }

    if (!runInference(image)) {
        qDebug() << "TensorFlow inference: ERROR";
        return {};
    }

    QList<double> regressors, classificators;

    if (!extractTensorData(classificators, regressors)) {
        qDebug() << "Tensor extraction: ERROR";
        return {};
    }

    QList<Anchor>    anchors    = extractSSDAnchors(anchorOptions_);
    QList<Detection> detections = extractDetections(faceOptions_, classificators, regressors, anchors);

    std::vector<Rect> ret;
    if (detections.length() == 0) {
        if (settings_.verbose) {
            qDebug() << "No face detected";
        }
        return ret;
    }

    if (settings_.verbose) {
        qDebug() << "Face detected";
    }

    for (auto det : detections) {
        auto  detect    = scaleDetection(det, image);
        QRect rectangle = QRectF(detect.xMin, detect.yMin, detect.width, detect.height).toAlignedRect();
        Point tl(rectangle.topLeft().x(), rectangle.topLeft().y());
        Point bt(rectangle.bottomRight().x(), rectangle.bottomRight().y());
        Rect  rect(tl, bt);
        ret.push_back(rect);
    }
    return ret;
}

QList<int> FaceDetection::getTensorShape(TfLiteTensor* tensor)
{
    QList<int>      shape;
    TfLiteIntArray* dims = tensor->dims;

    int size = dims->size;
    for (int ii_dim = 0; ii_dim < size; ii_dim++) {
        shape.push_back(dims->data[ii_dim]);
    }

    return shape;
}

bool FaceDetection::extractTensorData(QList<double>& outClass, QList<double>& outRegre)
{
    auto          tensorIdxclass = interpreter_->outputs()[0];
    TfLiteTensor* tensorClass    = interpreter_->tensor(tensorIdxclass);
    QList<int>    shapeClass     = getTensorShape(tensorClass);
    if (shapeClass.length() != 3) {
        return false;
    }
    auto sizeClass = shapeClass[0] * shapeClass[1] * shapeClass[2];

    auto          tensorIdxRegre = interpreter_->outputs()[1];
    TfLiteTensor* tensorRegre    = interpreter_->tensor(tensorIdxRegre);
    QList<int>    shapeRegre     = getTensorShape(tensorRegre);
    if (shapeRegre.length() != 3) {
        return false;
    }
    auto sizeRegre = shapeRegre[0] * shapeRegre[1] * shapeRegre[2];

    int indexClass = 0;
    int indexRegre = 1;
    if (sizeClass > sizeRegre) {
        indexClass = 1;
        indexRegre = 0;
        swap(tensorIdxclass, tensorIdxRegre);
        swap(sizeClass, sizeRegre);
    }

    switch (interpreter_->tensor(tensorIdxclass)->type) {
    case kTfLiteFloat32: {
        float* output = interpreter_->typed_output_tensor<float>(indexClass);
        if (output == nullptr) {
            return false;
        }

        for (int i = 0; i < sizeClass; i++) {
            outClass.append(static_cast<double>(output[i]));
        }

        break;
    }
    default: {
        qDebug() << "Cannot handle output " << interpreter_->tensor(tensorIdxclass)->type;
        qDebug() << "Classificators extraction: ERROR";
        return false;
    }
    }

    switch (interpreter_->tensor(tensorIdxRegre)->type) {
    case kTfLiteFloat32: {
        float* output = interpreter_->typed_output_tensor<float>(indexRegre);
        if (output == nullptr) {
            return false;
        }

        for (int i = 0; i < sizeRegre; i++) {
            outRegre.append(static_cast<double>(output[i]));
        }

        break;
    }
    default: {
        qDebug() << "Cannot handle output " << interpreter_->tensor(tensorIdxRegre)->type;
        qDebug() << "Regressors extraction: ERROR";
        return false;
    }
    }

    return true;
}

double FaceDetection::calScale(double min, double max, int strideIndex, int numStrides)
{
    if (numStrides == 1) {
        return (min + max) * 0.5;
    } else {
        return (max - min) * 1.0 * strideIndex / (numStrides - 1.0);
    }
}

QList<Anchor> FaceDetection::extractSSDAnchors(AnchorOptions options)
{
    // generate anchors
    QList<Anchor> anchors;
    int           strideSize = options.stridesSize();
    if (strideSize != options.numLayers) {
        qDebug() << "Anchors extraction options: ERROR";
        return anchors;
    }

    int layerId = 0;
    // We can have multiple layers, though the face detector only has one.
    while (layerId < options.stridesSize()) {
        QList<double> anchorHeight;
        QList<double> anchorWidth;
        QList<double> aspectRatios;
        QList<double> scales;

        int lastSameStrideLayer = layerId;
        while (lastSameStrideLayer < options.stridesSize() && options.strides[lastSameStrideLayer] == options.strides[layerId]) {
            double scale = calScale(options.minScale, options.maxScale, lastSameStrideLayer, options.stridesSize());
            if (lastSameStrideLayer == 0 && options.reduceBoxesInLowestLayer) {
                aspectRatios.append(1.0);
                aspectRatios.append(2.0);
                aspectRatios.append(0.5);
                scales.append(0.1);
                scales.append(scale);
                scales.append(scale);
            } else {
                for (int i = 0; i < options.aspectRatios.length(); i++) {
                    aspectRatios.append(options.aspectRatios[i]);
                    scales.append(scale);
                }

                if (options.interpolatedScaleAspectRatio > 0.0) {
                    double scaleNext = 0.0;
                    if (lastSameStrideLayer == options.stridesSize() - 1) {
                        scaleNext = 1.0;
                    } else {
                        scaleNext = calScale(options.minScale, options.maxScale, lastSameStrideLayer + 1, options.stridesSize());
                    }

                    scales.append(sqrt(scale * scaleNext));
                    aspectRatios.append(options.interpolatedScaleAspectRatio);
                }
            }

            lastSameStrideLayer++;
        }

        for (int i = 0; i < aspectRatios.length(); i++) {
            double ratioSQRT = sqrt(aspectRatios[i]);
            anchorHeight.append(scales[i] / ratioSQRT);
            anchorWidth.append(scales[i] * ratioSQRT);
        }

        int featureMapHeight = 0;
        int featureMapWidth  = 0;
        if (options.featureMapHeightSize() > 0) {
            featureMapHeight = options.featureMapHeight[layerId];
            featureMapWidth  = options.featureMapWidth[layerId];
        } else {
            int stride       = options.strides[layerId];
            featureMapHeight = ceil(1.0 * options.inputSizeHeight / stride);
            featureMapWidth  = ceil(1.0 * options.inputSizeWidth / stride);
        }

        // The core grid calculation is done here.
        for (int y = 0; y < featureMapHeight; y++) {
            for (int x = 0; x < featureMapWidth; x++) {
                for (int anchorId = 0; anchorId < anchorHeight.length(); anchorId++) {
                    double xCenterValue = (x + options.anchorOffsetX) * 1.0 / featureMapWidth;
                    double yCenterValue = (y + options.anchorOffsetY) * 1.0 / featureMapHeight;
                    double widthValue   = 0;
                    double heightValue  = 0;

                    if (options.fixedAnchorSize) {
                        widthValue  = 1.0;
                        heightValue = 1.0;
                    } else {
                        widthValue  = anchorWidth[anchorId];
                        heightValue = anchorHeight[anchorId];
                    }

                    Anchor anchor{
                        .xCenter = xCenterValue,
                        .yCenter = yCenterValue,
                        .h       = heightValue,
                        .w       = widthValue};

                    anchors.append(anchor);
                }
            }
        }

        layerId = lastSameStrideLayer;
    }

    return anchors;
}

QList<Detection> FaceDetection::extractDetections(FaceOptions options, QList<double> rawScores, QList<double> rawBoxes, QList<Anchor> anchors)
{
    QList<double> boxes = decodeBoxes(rawBoxes, anchors, options);
    QList<double> scores;
    QList<int>    classes;

    for (int i = 0; i < options.numBoxes; i++) {
        int    classId  = -1;
        double maxScore = 0.0;

        for (int scoreIdx = 0; scoreIdx < options.numClasses; scoreIdx++) {
            double score = rawScores[i * options.numClasses + scoreIdx];
            if (options.sigmoidScore) {
                if (isgreater(options.scoreClippingThresh, 0.0)) {
                    if (score < -options.scoreClippingThresh) {
                        score = -options.scoreClippingThresh;
                    }

                    if (score > options.scoreClippingThresh) {
                        score = options.scoreClippingThresh;
                    }

                    score = 1.0 / (1.0 + exp(-score));
                }
            }

            if (isless(maxScore, score)) {
                maxScore = score;
                classId  = scoreIdx;
            }
        }

        classes.append(classId);
        scores.append(maxScore);
    }

    QList<Detection> raw     = convertPropositionsToDetections(boxes, scores, classes, options);
    QList<Detection> refined = filterDetections(raw, options.overlapThresh);

    return refined;
}

QList<Detection> FaceDetection::convertPropositionsToDetections(QList<double> boxes, QList<double> scores, QList<int> classes, FaceOptions options)
{
    QList<Detection> outputDetections;

    for (int i = 0; i < options.numBoxes; i++) {
        if (isless(scores[i], options.minScoreThresh)) {
            continue;
        }

        int boxOffset = i * options.numCoords;

        Detection detection = convertPropositionToDetection(
            boxes[boxOffset + 0],
            boxes[boxOffset + 1],
            boxes[boxOffset + 2],
            boxes[boxOffset + 3],
            boxes[boxOffset + 4],
            boxes[boxOffset + 5],
            boxes[boxOffset + 6],
            boxes[boxOffset + 7],
            boxes[boxOffset + 8],
            boxes[boxOffset + 9],
            boxes[boxOffset + 10],
            boxes[boxOffset + 11],
            boxes[boxOffset + 12],
            boxes[boxOffset + 13],
            boxes[boxOffset + 14],
            boxes[boxOffset + 15],
            scores[i],
            classes[i],
            options.flipVertically);

        outputDetections.append(detection);
    }

    return outputDetections;
}

Detection FaceDetection::convertPropositionToDetection(double boxYMin, double boxXMin, double boxYMax, double boxXMax, double xLeftEye, double yLeftEye, double xRightEye, double yRightEye, double xNose, double yNose, double xMouth, double yMouth, double xLeftEar, double yLeftEar, double xRightEar, double yRightEar, double score, int classId, bool flipVertically)
{
    double yMin;

    if (flipVertically) {
        yMin = 1.0 - boxYMax;
    } else {
        yMin = boxYMin;
    }

    Detection detection{
        .classId   = classId,
        .height    = (boxYMax - boxYMin),
        .width     = (boxXMax - boxXMin),
        .score     = score,
        .xMin      = boxXMin,
        .yMin      = yMin,
        .xLeftEye  = xLeftEye,
        .yLeftEye  = yLeftEye,
        .xRightEye = xRightEye,
        .yRightEye = yRightEye,
        .xNose     = xNose,
        .yNose     = yNose,
        .xMouth    = xMouth,
        .yMouth    = yMouth,
        .xLeftEar  = xLeftEar,
        .yLeftEar  = yLeftEar,
        .xRightEar = xRightEar,
        .yRightEar = yRightEar};

    return detection;
}

QList<double> FaceDetection::decodeBoxes(QList<double> rawBoxes, QList<Anchor> anchors, FaceOptions options)
{
    QList<double> boxes;

    for (int i = 0; i < options.numBoxes; i++) {
        int boxOffset = i * options.numCoords + options.boxCoordOffset;

        double yCenter = rawBoxes[boxOffset];
        double xCenter = rawBoxes[boxOffset + 1];

        double h = rawBoxes[boxOffset + 2];
        double w = rawBoxes[boxOffset + 3];

        if (options.reverseOutputOrder) {
            xCenter = rawBoxes[boxOffset];
            yCenter = rawBoxes[boxOffset + 1];

            w = rawBoxes[boxOffset + 2];
            h = rawBoxes[boxOffset + 3];
        }

        xCenter = xCenter / options.xScale * anchors[i].w + anchors[i].xCenter;
        yCenter = yCenter / options.yScale * anchors[i].h + anchors[i].yCenter;

        if (options.applyExponentialOnBoxSize) {
            h = exp(h / options.hScale) * anchors[i].h;
            w = exp(w / options.wScale) * anchors[i].w;
        } else {
            h = h / options.hScale * anchors[i].h;
            w = w / options.wScale * anchors[i].w;
        }

        double yMin = yCenter - h / 2.0;
        double xMin = xCenter - w / 2.0;
        double yMax = yCenter + h / 2.0;
        double xMax = xCenter + w / 2.0;

        boxes.push_back(yMin);
        boxes.push_back(xMin);
        boxes.push_back(yMax);
        boxes.push_back(xMax);

        for (int j = 0; j < options.numKeyPoints; j++) {
            int keypointOffset = i * options.numCoords + options.keyPointCoordOffset + j * options.numValuesPerKeypoint;

            double keyPointY = rawBoxes[keypointOffset];
            double keyPointX = rawBoxes[keypointOffset + 1];

            if (options.reverseOutputOrder) {
                keyPointX = rawBoxes[keypointOffset];
                keyPointY = rawBoxes[keypointOffset + 1];
            }

            boxes.push_back(keyPointX / options.xScale * anchors[i].w + anchors[i].xCenter);
            boxes.push_back(keyPointY / options.yScale * anchors[i].h + anchors[i].yCenter);
        }
    }

    return boxes;
}

QList<Detection> FaceDetection::filterDetections(QList<Detection> detections, double threshold)
{
    QList<Detection> filtered;
    auto             it = detections.constBegin();
    while (it != detections.constEnd()) {
        auto itt = it + 1;
        while (itt != detections.constEnd()) {
            auto iRect = QRectF(it->xMin * inputImageWidth_, it->yMin * inputImageHeight_, it->width * inputImageWidth_, it->height * inputImageHeight_).toAlignedRect();
            auto jRect = QRectF(itt->xMin * inputImageWidth_, itt->yMin * inputImageHeight_, itt->width * inputImageWidth_, itt->height * inputImageHeight_).toAlignedRect();

            auto intersectRect = (jRect & iRect);
            auto interArea     = intersectRect.height() * intersectRect.width();
            auto iRectArea     = iRect.height() * iRect.width();

            if (interArea >= iRectArea * threshold)
                break;
            itt++;
        }
        if (itt == detections.constEnd()) {
            filtered.append(*it);
        }
        it++;
    }
    return filtered;
}

QList<double> FaceDetection::quickSort(QList<double> a)
{
    if (a.length() <= 1)
        return a;

    double        pivot = a[0];
    QList<double> less;
    QList<double> more;
    QList<double> pivotList;

    for (double i : a) {
        if (isless(i, pivot)) {
            less.append(i);
        } else if (isgreater(i, pivot)) {
            more.append(i);
        } else {
            pivotList.append(i);
        }
    }

    less = quickSort(less);
    more = quickSort(more);

    less.append(pivotList);
    less.append(more);

    return less;
}

QList<double> FaceDetection::sum(double a, QList<double> b)
{
    QList<double> temp;

    for (double element : b) {
        temp.append(a + element);
    }

    return temp;
}

QList<double> FaceDetection::maximum(double value, QList<double> itemIndex)
{
    QList<double> temp;

    for (double element : itemIndex) {
        if (isgreater(value, element)) {
            temp.append(value);
        } else {
            temp.append(element);
        }
    }

    return temp;
}

QList<double> FaceDetection::itemIndex(QList<double> items, QList<int> positions)
{
    QList<double> temp;

    for (int position : positions) {
        temp.append(items[position]);
    }

    return temp;
}

QList<double> FaceDetection::subtract(QList<double> a, QList<double> b)
{
    QList<double> temp;
    if (a.length() != b.length())
        return temp;

    for (int i = 0; i < a.length(); i++) {
        temp.append(a[i] - b[i]);
    }

    return temp;
}

QList<double> FaceDetection::multiply(QList<double> a, QList<double> b)
{
    QList<double> temp;
    if (a.length() != b.length())
        return temp;

    for (int i = 0; i < a.length(); i++) {
        temp.append(a[i] * b[i]);
    }

    return temp;
}

QList<double> FaceDetection::divide(QList<double> a, QList<double> b)
{
    QList<double> temp;
    if (a.length() != b.length())
        return temp;

    for (int i = 0; i < a.length(); i++) {
        temp.append(a[i] / b[i]);
    }

    return temp;
}

Detection FaceDetection::scaleDetection(Detection detection, const QImage& image)
{
    Detection scaled = detection;

    qreal width  = image.width();
    qreal height = image.height();

    scaled.width     = detection.width * width;
    scaled.height    = detection.height * height;
    scaled.xMin      = detection.xMin * width;
    scaled.yMin      = detection.yMin * height;
    scaled.xLeftEye  = detection.xLeftEye * width;
    scaled.yLeftEye  = detection.yLeftEye * height;
    scaled.xRightEye = detection.xRightEye * width;
    scaled.yRightEye = detection.yRightEye * height;
    scaled.xLeftEar  = detection.xLeftEar * width;
    scaled.yLeftEar  = detection.yLeftEar * height;
    scaled.xRightEar = detection.xRightEar * width;
    scaled.yRightEar = detection.yRightEar * height;
    scaled.xNose     = detection.xNose * width;
    scaled.yNose     = detection.yNose * height;
    scaled.xMouth    = detection.xMouth * width;
    scaled.yMouth    = detection.yMouth * height;

    return scaled;
}
