#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QGraphicsScene>
#include <qgraphicsview.h>
#include <qdebug.h>
#include <QVector2D>
#include <QFuture>
#include <QtConcurrent/QtConcurrentRun>
#include <algorithm>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->button_load_image, SIGNAL(clicked()), this, SLOT(button_load_clicked()));

    connect(ui->button_filter_sobel, SIGNAL(clicked()), this, SLOT(button_sobel_clicked()));

    connect(ui->button_filter_madian, SIGNAL(clicked()), this, SLOT(button_median_clicked()));

    ui->label4output->setMinimumSize(300, 300);
    ui->label4output->setMaximumSize(300, 300);
}

void MainWindow::button_load_clicked() {

    QString file_name = QFileDialog::getOpenFileName(this, "Open File",
                                                    "/home/stra10/Desktop/", //TODO NAPRAVI OVO LEPO DA RADI SVUDA
                                                    "Images (*.png)");
    if(nullptr == file_name)
    {
        qDebug() << "Nisi dao putanju do fajla, pucam ovde";
    }

    imp.load_image(file_name);

    set_image();

}

void MainWindow::button_sobel_clicked() {

    QPixmap qpm;
    QImage qim;
    qim = imp.get_image();

    qim = toGrayscale(qim);

    qim = applySobelFilter(qim);

    qpm.convertFromImage(qim);

    int image_width  = ui->label4output->width();
    int image_height = ui->label4output->height();

    ui->label4output->setPixmap(qpm.scaled(image_width, image_height, Qt::KeepAspectRatio));
}

void MainWindow::button_median_clicked() {
    QPixmap qpm;
    QImage qim;
    qim = imp.get_image();

    qim = applyMedianFilter(qim);

    qpm.convertFromImage(qim);

    int image_width  = ui->label4output->width();
    int image_height = ui->label4output->height();

    ui->label4output->setPixmap(qpm.scaled(image_width, image_height, Qt::KeepAspectRatio));

}


void MainWindow::set_image() // shows the image in LabeL_image space
{
    QPixmap qpm;
    qpm.convertFromImage(imp.get_image());

    int image_width  = ui->label_image->width();
    int image_height = ui->label_image->height();

    ui->label_image->setPixmap(qpm.scaled(image_width, image_height, Qt::KeepAspectRatio));

}


MainWindow::~MainWindow()
{
    delete ui;
}

QImage &MainWindow::toGrayscale(QImage &qim)
{
    for (int i = 0; i < qim.height(); i++)
    {
        int depth =4;
        uchar* scan = qim.scanLine(i);
        for (int j = 0; j < qim.width(); j++) {
            QRgb* rgbpixel = reinterpret_cast<QRgb*>(scan + j*depth);
            int gray = qGray(*rgbpixel);
            *rgbpixel = QColor(gray, gray, gray).rgba();
        }
    }

    return qim;
}

QImage &MainWindow::applySobelFilter(QImage &qim) {
    if(!(qim.isGrayscale()))
    {
        qDebug() << "The QImage I got here is not a grayscale image, so there is no way I'm doing the Sobel!" ;
        return qim;
    }

    QImage qim_copy = qim;

    uchar* scan_previous = qim_copy.scanLine(0);
    uchar* scan_current = qim_copy.scanLine(1);
    uchar* scan_next = qim_copy.scanLine(2);

    QList<QFuture<void> > futures;
    for (int i = 1; i < qim.height() - 1; i++)
    {
        uchar* output_scan_current = qim.scanLine(i);

        auto future = QtConcurrent::run(MainWindow::calculateLineValueSobel, qim.width() - 1,
                                             output_scan_current, scan_previous, scan_current, scan_next);

        futures.append(future);

        scan_previous = scan_current;
        scan_current = scan_next;
        scan_next = qim_copy.scanLine(i+2);
    }

    for(auto future : futures) {
        future.waitForFinished();
    }

    return qim;
}

int MainWindow::calculateLineValueSobel(int width, uchar* output_scan_current,
                                         uchar* scan_previous, uchar* scan_current, uchar* scan_next)
{
    for (int j = 1; j < width; ++j) {

        int depth = 4;

        QVector<QVector<int>> sobel_horizontal = {{ -1,  -2,  -1},
                                                  {  0,   0,   0},
                                                  {  1,   2,   1}};

        QVector<QVector<int>> sobel_vertical   = {{ -1,  0,  1},
                                                   {-2,  0,  2},
                                                   {-1,  0,  1}};

        QVector<QVector<int>> tmp;
                              tmp =  {{ 0,  0,  0},
                                      { 0,  0,  0},
                                      { 0,  0,  0}};

        QRgb* upper_left    = reinterpret_cast<QRgb*>(scan_previous + (j-1)*depth);
        QRgb* upper_middle  = reinterpret_cast<QRgb*>(scan_previous + (j  )*depth);
        QRgb* upper_right   = reinterpret_cast<QRgb*>(scan_previous + (j+1)*depth);
        QRgb* center_left   = reinterpret_cast<QRgb*>(scan_current  + (j-1)*depth);
        QRgb* current       = reinterpret_cast<QRgb*>(scan_current  + (j  )*depth);
        QRgb* center_right  = reinterpret_cast<QRgb*>(scan_current  + (j+1)*depth);
        QRgb* bottom_left   = reinterpret_cast<QRgb*>(scan_next     + (j-1)*depth);
        QRgb* bottom_middle = reinterpret_cast<QRgb*>(scan_next     + (j  )*depth);
        QRgb* bottom_right  = reinterpret_cast<QRgb*>(scan_next     + (j+1)*depth);

        QRgb* output_current = reinterpret_cast<QRgb*>(output_scan_current  + (j  )*depth);

        tmp[0][0] = qRed(*upper_left); //qRed da bi dohvatio prvu koordinatu/boju, a sve su iste posto je vec grayscale
        tmp[0][1] = qRed(*upper_middle);
        tmp[0][2] = qRed(*upper_right);
        tmp[1][0] = qRed(*center_left);
        tmp[1][1] = qRed(*current);
        tmp[1][2] = qRed(*center_right);
        tmp[2][0] = qRed(*bottom_left);
        tmp[2][1] = qRed(*bottom_middle);
        tmp[2][2] = qRed(*bottom_right);

        //postavili smo tmp matricu na svoje vrednosti, jos samo da izracunamo
        //sta da stavimo na current position i to je to

        int sum_h = 0;
        int sum_v = 0;

        for(int p = 0; p < 3; p++)
        {
            for(int q = 0; q < 3; q++)
            {
                sum_h += tmp[p][q] * sobel_horizontal[p][q];
                sum_v += tmp[p][q] * sobel_vertical[p][q];
            }
        }
        int sum = sum_h + sum_v;
        // za svaki slucaj, mada ne bi trebalo da upadnemo ovde ikad -- izgleda da upadamo prilicno cesto
        if(sum > 255)
            sum = 255;
        if(sum < 0)
            sum = 0;

        *output_current = QColor(sum, sum, sum).rgba();

    }

    return 0;
}

QImage &MainWindow::applyMedianFilter(QImage &qim)
{
    QImage qim_copy = qim;

    uchar* scan_previous = qim_copy.scanLine(0);
    uchar* scan_current  = qim_copy.scanLine(1);
    uchar* scan_next     = qim_copy.scanLine(2);

    QList<QFuture<void> > futures;

    for (int i = 1; i < qim.height() - 1; i++)
    {
        uchar* output_scan_current = qim.scanLine(i);

        //funkcija pravi novi thread svaki put kad je pozvana..
        //u teoriji bi trebalo da dobijemo neko ubrzanje......
        auto future = QtConcurrent::run(MainWindow::calculateLineValueMedian, qim.width()-1,
                           output_scan_current, scan_previous, scan_current, scan_next);

        futures.append(future);

        scan_previous = scan_current;
        scan_current  = scan_next;
        scan_next     = qim_copy.scanLine(i+2);
    }

    for(auto future : futures) {
        future.waitForFinished();
    }
    return qim;
}

int MainWindow::calculateLineValueMedian(int width, uchar* output_scan_current, uchar* scan_previous, uchar* scan_current, uchar* scan_next)
{

    for (int j = 0; j < width; ++j) {

        int depth = 4;

        QRgb* center_left    = reinterpret_cast<QRgb*>(scan_current  + (j-1)*depth);
        QRgb* current        = reinterpret_cast<QRgb*>(scan_current  + (j  )*depth);
        QRgb* center_right   = reinterpret_cast<QRgb*>(scan_current  + (j+1)*depth);

        QRgb* output_current = reinterpret_cast<QRgb*>(output_scan_current  + (j  )*depth);

        QVector<int> tmp(8);

        for (int i = -1; i < 2; ++i)
            tmp[i+1] = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
        tmp[3] = qRed(*center_left);
        tmp[4] = qRed(*center_right);
        for (int i = -1; i < 2; ++i)
            tmp[5+i+1] = qRed(*reinterpret_cast<QRgb*>(scan_next + (j+i)*depth));

        std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
        auto r = tmp[tmp.size()/2];

        for (int i = -1; i < 2; ++i)
            tmp[i+1] = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
        tmp[3] = qGreen(*center_left);
        tmp[4] = qGreen(*center_right);
        for (int i = -1; i < 2; ++i)
            tmp[5+i+1] = qGreen(*reinterpret_cast<QRgb*>(scan_next + (j+i)*depth));

        std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
        auto g = tmp[tmp.size()/2];

        for (int i = -1; i < 2; ++i)
            tmp[i+1] = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
        tmp[3] = qBlue(*center_left);
        tmp[4] = qBlue(*center_right);
        for (int i = -1; i < 2; ++i)
            tmp[5+i+1] = qBlue(*reinterpret_cast<QRgb*>(scan_next + (j+i)*depth));

        std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
        auto b = tmp[tmp.size()/2];

        auto a = qAlpha(*current);
        *output_current = QColor(r, g, b, a).rgba();
    }

    return 0;
}

