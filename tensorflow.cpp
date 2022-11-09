#include "tensorflow.h"
#include <QDebug>
#include <QDir>
#include <QString>

TensorFlow::TensorFlow()
{}

TensorFlow::~TensorFlow()
{
    interpreter_.reset();
    model_.reset();
}

// WORKFLOW
bool TensorFlow::initialize(int height, int width, QString modelPath)
{
    if (modelPath.isEmpty()) {
        return false;
    }

    modelPath_ = modelPath;
    model_     = FlatBufferModel::BuildFromFile(modelPath_.toStdString().c_str(), &reporter_);

    if (model_ == nullptr) {
        qDebug() << "TensorFlow model loading: ERROR";
        return false;
    }

    inputImageHeight_ = height;
    inputImageWidth_  = width;

    try {
        InterpreterBuilder builder(*model_.get(), resolver_);
        if (builder(&interpreter_) != kTfLiteOk) {
            qDebug() << "TensorFlow interpreter check: ERROR";
            return false;
        }

        if (settings_.acceleration) {
            auto delegates_ = getDelegates();

            for (const auto& delegate : delegates_) {
                if (interpreter_->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk) {
                    qDebug() << "Failed to apply " << delegate.first.c_str() << " delegate.\n";
                } else {
                    qDebug() << "Applied " << delegate.first.c_str() << " delegate.\n";
                }
            }
        }

        if (interpreter_->AllocateTensors() != kTfLiteOk) {
            qDebug() << "TensorFlow allocate tensors: ERROR";
        }

        int input = interpreter_->inputs()[0];

        TfLiteIntArray* dims = interpreter_->tensor(input)->dims;
        targetedImageHeight_ = dims->data[1];
        targetedImageWidth_  = dims->data[2];

        interpreter_->UseNNAPI(false);
        interpreter_->SetAllowFp16PrecisionForFp32(true);

        if (settings_.verbose) {
            qDebug() << "Wanted height:" << targetedImageHeight_;
            qDebug() << "Wanted width:" << targetedImageWidth_;

            qDebug() << "Print current interpreter state...";
            PrintInterpreterState(interpreter_.get());
        }

        if (settings_.threads > 1) {
            interpreter_->SetNumThreads(settings_.threads);
        }
        return true;
    } catch (...) {
        qDebug() << "TensorFlow loading model: ERROR";
        return false;
    }
}

bool TensorFlow::setInputTensor(const QImage& image)
{
    std::vector<int> inputs = interpreter_->inputs();

    for (unsigned int i = 0; i < interpreter_->inputs().size(); i++) {
        int input = inputs[i];

        switch (interpreter_->tensor(input)->type) {
        case kTfLiteFloat32: {
            initInputTensor<float>(
                interpreter_->typed_tensor<float>(input), image,
                targetedImageHeight_, targetedImageWidth_);

            break;
        }
        case kTfLiteUInt8: {
            initInputTensor<uint8_t>(
                interpreter_->typed_tensor<uint8_t>(input), image,
                targetedImageHeight_, targetedImageWidth_);

            break;
        }
        default: {
            qDebug() << "Cannot handle input type" << interpreter_->tensor(input)->type << "yet";
            return false;
        }
        }
    }

    return true;
}

bool TensorFlow::runInference(const QImage& image)
{
    if (!initialized_) {
        qDebug() << "TensorFlow run pipeline: ERROR";
        return false;
    }

    if (!setInputTensor(image)) {
        qDebug() << "TensorFlow set inputs: ERROR";
        return false;
    }

    if (interpreter_->Invoke() != kTfLiteOk) {
        qDebug() << "TensorFlow invoke interpreter: ERROR";
        return false;
    }

    return true;
}

// ACCELERATION
TfLiteDelegatePtr TensorFlow::createNNAPIDelegate()
{
    return TfLiteDelegatePtr(tflite::NnApiDelegate(), [](TfLiteDelegate*) {});
}

TfLiteDelegatePtrMap TensorFlow::getDelegates()
{
    TfLiteDelegatePtrMap delegates;
    TfLiteDelegatePtr    NNAPIDelegate = createNNAPIDelegate();

    if (!NNAPIDelegate) {
        qDebug() << "NNAPI acceleration is unsupported on this platform.";
    } else {
        delegates.emplace("NNAPI", std::move(NNAPIDelegate));
    }

    return delegates;
}

// SETTINGS
void TensorFlow::setUseInferenceHardwareDelegates(bool value)
{
    settings_.acceleration = value;

    if (settings_.acceleration) {
        qDebug() << "TensorFlow activating hardware acceleration...";
    } else {
        qDebug() << "TensorFlow deactivating hardware acceleration...";
    }

    if (inputImageHeight_ > 0 && inputImageWidth_ > 0) {
        initialize(inputImageHeight_, inputImageWidth_, modelPath_);
    }
}

void TensorFlow::setObjectDetectionThreshold(double value)
{
    settings_.detectionThreshold = value;
}

void TensorFlow::setInferenceThreads(int value)
{
    settings_.threads = value;
}

void TensorFlow::setVerbose(bool value)
{
    settings_.verbose = value;
}
