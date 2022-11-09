#include "calibration.h"

using namespace std;
using namespace cv;

StereoCalibrator::StereoCalibrator(const QString& leftImageNamePattern, const QString& rightImageNamePattern, const QString& imageDirRelPath, const QString& cameraDataRelPath, const QString& leftCameraDataFileName, const QString& rightCameraDataFileName, const QString& stereoCameraDataFileName, int chessBoardWidth, int chessBoardHeigth, int chessBoardSquareSize)
    : leftImageNamePattern_(leftImageNamePattern)
    , rightImageNamePattern_(rightImageNamePattern)
    , imageDirRelPath_(imageDirRelPath)
    , cameraDataRelPath_(cameraDataRelPath)
    , leftCameraDataFileName_(leftCameraDataFileName)
    , rightCameraDataFileName_(rightCameraDataFileName)
    , stereoCameraDataFileName_(stereoCameraDataFileName)
{
    chessBoard_.width      = chessBoardWidth;
    chessBoard_.height     = chessBoardHeigth;
    chessBoard_.squareSize = chessBoardSquareSize; //mm
}

std::optional<IntrinsicData> StereoCalibrator::calibrationIntrinsic(const ChessBoardData& chessBoard, const QString& imageDir, const QString& imageNamePattern, const QString& cameraDir, const QString& cameraDataName)
{
    vector<Point2f>         corners;
    vector<vector<Point3f>> cornersRealCoordinates;
    vector<vector<Point2f>> cornersImageCoordinates;
    QString                 imagePath;

    // locate images for calibration
    QDir       directory(imageDir);
    const auto absolutePath = directory.absolutePath();
    const auto images       = directory.entryList(QStringList() << imageNamePattern, QDir::Files);

    // get all points for calibration
    for (const auto& imagename : images) {
        imagePath = absolutePath + QDir::separator() + imagename;
        auto img  = imread(imagePath.toStdString(), IMREAD_COLOR);
        Mat  gray;
        cvtColor(img, gray, CV_RGB2GRAY);

        Size boardSize = Size(chessBoard.width, chessBoard.height);
        auto found     = cv::findChessboardCorners(img, boardSize, corners);

        if (found) {
            cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
                         TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001)); //maxCount=40 and epsilon=0.001
            drawChessboardCorners(gray, boardSize, corners, found);
            auto message = QString("Calibrate image: ") + imagename;
            qInfo() << message;
            vector<Point3f> cornerReal;
            for (int i = 0; i < chessBoard.height; i++)
                for (int j = 0; j < chessBoard.width; j++)
                    cornerReal.push_back(
                        Point3f(static_cast<double>(j) * chessBoard.squareSize, static_cast<double>(i) * chessBoard.squareSize, 0));

            cornersImageCoordinates.push_back(corners);
            cornersRealCoordinates.push_back(cornerReal);

        } else {
            auto message = imagename + QString(" can not find chessboard!");
            qWarning() << message;
        }
    }
    if (cornersImageCoordinates.empty()) {
        return {};
    }

    // calibration
    Mat         K;
    Mat         D;
    vector<Mat> rvecs;
    vector<Mat> tvecs;

    int  flag = 0; //CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6;
    auto imageSize
        = imread(imagePath.toStdString(), IMREAD_COLOR).size();
    calibrateCamera(cornersRealCoordinates, cornersImageCoordinates, imageSize, K, D, rvecs, tvecs,
                    flag);

    // save camera data
    if (!QDir(cameraDir).exists()) {
        QDir().mkdir(cameraDir);
    }
    QDir        saveDir(cameraDir);
    const auto  cameraDataPath = saveDir.absolutePath() + QDir::separator() + cameraDataName;
    FileStorage intrinsicData(cameraDataPath.toStdString(), FileStorage::WRITE);
    intrinsicData << "K" << K;
    intrinsicData << "D" << D;
    auto message = QString("Save camera data in file ") + cameraDataPath;
    qInfo() << message;
    intrinsicData.release();

    IntrinsicData ret;
    ret.K = K;
    ret.D = D;
    return ret;
}

bool StereoCalibrator::calibrationStereo(const ChessBoardData& chessBoard, const QString& imageDir, const QString& imageNamePatternL, const QString& imageNamePatternR, const QString& cameraDir, const QString& cameraDataName, const IntrinsicData& intrinsicL, const IntrinsicData& intrinsicR)
{
    vector<Point2f>         cornersL;
    vector<Point2f>         cornersR;
    vector<vector<Point3f>> cornersRealCoordinates;
    vector<vector<Point2f>> cornersImageCoordinatesL;
    vector<vector<Point2f>> cornersImageCoordinatesR;
    QString                 imagePathL;
    QString                 imagePathR;
    const QRegExp           rx(QLatin1Literal("[^0-9]+")); // Regexp used to check if left anf right image are a pair

    // locate images for calibration
    QDir       directory(imageDir);
    const auto absolutePath = directory.absolutePath();
    const auto imagesL      = directory.entryList(QStringList() << imageNamePatternL, QDir::Files);
    const auto imagesR      = directory.entryList(QStringList() << imageNamePatternR, QDir::Files);

    // get all points for calibration, the number of left and right images should be identical
    auto imageNum = imagesL.size();
    qInfo() << "Find " << imageNum << " images";
    for (auto i = 0; i < imageNum; i++) {
        // check if the left and right are one pair
        const auto&& indexL = imagesL[i].split(rx, QString::SkipEmptyParts);
        const auto&& indexR = imagesR[i].split(rx, QString::SkipEmptyParts);
        QString      msg    = QString("find images ") + imagesL[i] + QString(" and ") + imagesR[i];
        if (indexL != indexR) {
            msg += QString(", but they are not a pair");
            qWarning() << msg;
            continue;
        }

        imagePathL = absolutePath + QDir::separator() + imagesL[i];
        imagePathR = absolutePath + QDir::separator() + imagesR[i];
        auto imgL  = imread(imagePathL.toStdString(), IMREAD_COLOR);
        auto imgR  = imread(imagePathR.toStdString(), IMREAD_COLOR);
        Mat  grayL, grayR;
        cvtColor(imgL, grayL, CV_RGB2GRAY);
        cvtColor(imgR, grayR, CV_RGB2GRAY);
        Size boardSize = Size(chessBoard.width, chessBoard.height);
        auto foundL    = cv::findChessboardCorners(imgL, boardSize, cornersL);
        auto foundR    = cv::findChessboardCorners(imgR, boardSize, cornersR);

        if (foundL && foundR) {
            msg += QString(", for calibration");
            qWarning() << msg;
            cornerSubPix(grayL, cornersL, cv::Size(5, 5), cv::Size(-1, -1),
                         TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001)); //maxCount=40 and epsilon=0.001
            cornerSubPix(grayR, cornersR, cv::Size(5, 5), cv::Size(-1, -1),
                         TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001)); //maxCount=40 and epsilon=0.001

            vector<Point3f> cornerReal;
            for (int i = 0; i < chessBoard.height; i++)
                for (int j = 0; j < chessBoard.width; j++)
                    cornerReal.push_back(
                        Point3f(static_cast<double>(j) * chessBoard.squareSize, static_cast<double>(i) * chessBoard.squareSize, 0));

            cornersImageCoordinatesL.push_back(cornersL);
            cornersImageCoordinatesR.push_back(cornersR);
            cornersRealCoordinates.push_back(cornerReal);
        } else {
            msg += QString(", but can not find corner!");
            qWarning() << msg;
        }
    }
    if (cornersRealCoordinates.empty()) {
        qWarning() << "No valid photos for stereo calibration";
        return false;
    }

    // calibration
    Mat   R, F, E;
    Vec3d T;

    Mat  KL   = intrinsicL.K;
    Mat  KR   = intrinsicR.K;
    Mat  DL   = intrinsicL.D;
    Mat  DR   = intrinsicR.D;
    auto flag = CALIB_FIX_INTRINSIC;
    auto imageSize
        = imread(imagePathL.toStdString(), IMREAD_COLOR).size();
    stereoCalibrate(cornersRealCoordinates, cornersImageCoordinatesL, cornersImageCoordinatesR, KL, DL, KR,
                    DR, imageSize, R, T, E, F, flag);

    // Rectification
    qInfo() << QString("StartRectification");
    Mat RL, RR, PL, PR, Q;
    stereoRectify(KL, DL, KR, DR, imageSize, R, T, RL, RR, PL, PR, Q);

    // save camera data
    if (!QDir(cameraDir).exists()) {
        QDir().mkdir(cameraDir);
    }
    QDir            saveDir(cameraDir);
    const auto      cameraDataPath = saveDir.absolutePath() + QDir::separator() + cameraDataName;
    cv::FileStorage stereoData(cameraDataPath.toStdString(), cv::FileStorage::WRITE);
    stereoData << "KL" << KL;
    stereoData << "KR" << KR;
    stereoData << "DL" << DL;
    stereoData << "DR" << DR;
    stereoData << "R" << R;
    stereoData << "T" << T;
    stereoData << "E" << E;
    stereoData << "F" << F;
    stereoData << "RL" << RL;
    stereoData << "RR" << RR;
    stereoData << "PL" << PL;
    stereoData << "PR" << PR;
    stereoData << "Q" << Q;
    qInfo() << QString("Save camera data in file ") + cameraDataPath;

    stereoData.release();
    return true;
}

void StereoCalibrator::calibrate()
{
    auto IntrinsicDataL = calibrationIntrinsic(chessBoard_, imageDirRelPath_ + QString("/"), leftImageNamePattern_, cameraDataRelPath_, leftCameraDataFileName_);
    auto IntrinsicDataR = calibrationIntrinsic(chessBoard_, imageDirRelPath_ + QString("/"), rightImageNamePattern_, cameraDataRelPath_, rightCameraDataFileName_);
    if (IntrinsicDataL.has_value() && IntrinsicDataR.has_value()) {
        auto ret = calibrationStereo(chessBoard_, imageDirRelPath_, leftImageNamePattern_, rightImageNamePattern_, cameraDataRelPath_, stereoCameraDataFileName_, IntrinsicDataL.value(), IntrinsicDataR.value());
        if (ret) {
            emit calibrationResult(0);
        } else {
            emit calibrationResult(1);
        }
    } else {
        emit calibrationResult(2);
    }
}
