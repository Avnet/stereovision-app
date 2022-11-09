#pragma once

#include <QDebug>
#include <QImage>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "auxutils.h"

using namespace cv;

using namespace tflite;
using namespace tflite::ops::builtin;

using TfLiteDelegatePtr    = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

struct Settings
{
    int  threads = 1;
    bool verbose = false;
#ifdef ARCH_PC
    bool acceleration = false;
#else
    bool acceleration = true;
#endif
    float detectionThreshold = 0.75;
};

class TensorFlow : public QObject
{
    Q_OBJECT
public:
    TensorFlow();
    ~TensorFlow();
    TensorFlow(TensorFlow const&) = delete;
    TensorFlow operator=(TensorFlow const&) = delete;

    static const int FACE_MEASUREMENTS = 4068;

    // WORKFLOW
    virtual bool initialize(int imageHeight, int imageWidth, QString modelPath);
    bool         setInputTensor(const QImage& image);
    bool         runInference(const QImage& image);

    // ACCELERATION
    TfLiteDelegatePtr    createNNAPIDelegate();
    TfLiteDelegatePtrMap getDelegates();

    // SETTINGS
    void setUseInferenceHardwareDelegates(bool value);
    void setObjectDetectionThreshold(double value);
    void setInferenceThreads(int value);
    void setVerbose(bool value);

protected:
    std::unique_ptr<Interpreter>     interpreter_;
    std::unique_ptr<FlatBufferModel> model_;

    BuiltinOpResolver resolver_;
    StderrReporter    reporter_;
    Settings          settings_;
    QString           modelPath_;

    int targetedImageHeight_ = 0;
    int targetedImageWidth_  = 0;

    int inputImageHeight_ = 0;
    int inputImageWidth_  = 0;

    bool initialized_ = false;
};

template<class T>
void initInputTensor(T* out, const QImage& in, int targetedImageHeight, int targetedImageWidth)
{
    if (in.isNull()) {
        qDebug() << "TensorFlow input tensors initilization: ERROR";
        exit(-1);
    }

    Mat matrix = AuxUtils::convertQImageToMat(in);

    resize(matrix, matrix, Size(targetedImageWidth, targetedImageHeight), 0, 0, INTER_LINEAR);

    T* first = out;

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            Vec4b pixel = matrix.at<Vec4b>(i, j);

            float red   = (pixel.val[2] - 127.5) / 127.5;
            float green = (pixel.val[1] - 127.5) / 127.5;
            float blue  = (pixel.val[0] - 127.5) / 127.5;

            *out = static_cast<T>(red);
            out++;

            *out = static_cast<T>(green);
            out++;

            *out = static_cast<T>(blue);
            out++;
        }
    }

    out = first;
}
