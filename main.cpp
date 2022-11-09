#include <QApplication>
#include "mainwindow.h"

int main(int argc, char* argv[])
{
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<std::vector<Rect>>("std::vector<Rect>");

    QApplication a(argc, argv);
    MainWindow   w;
    w.showMaximized();

    return a.exec();
}
