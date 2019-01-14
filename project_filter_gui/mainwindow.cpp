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
#include <caffe/caffe.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "./src/net.hpp"

#define SOBEL       (0)
#define MEDIAN      (1)
#define GAUSS       (2)
#define EMBOSS      (3)
#define SHARPENING  (4)
#define BLUR        (5)
int current_filter;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->button_load_image, SIGNAL(clicked()), this, SLOT(button_load_clicked()));

    connect(ui->button_filter_sobel, SIGNAL(clicked()), this, SLOT(button_sobel_clicked()));

    connect(ui->button_filter_madian, SIGNAL(clicked()), this, SLOT(button_median_clicked()));

    connect(ui->button_filter_sharpening, SIGNAL(clicked()), this, SLOT(button_sharpening_clicked()));

    connect(ui->button_filter_blur, SIGNAL(clicked()), this, SLOT(button_blur_clicked()));

    connect(ui->button_filter_emboss, SIGNAL(clicked()), this, SLOT(button_emboss_clicked()));

    ui->label4output->setMinimumSize(300, 300);

    ui->label_image->setMinimumSize(300, 300);

    ui->centralWidget->setMinimumSize(700,400);
}

void MainWindow::button_load_clicked() {

    QString file_name = QFileDialog::getOpenFileName(this, "Open File",
                                                    "../../../../Pictures", //TODO NAPRAVI OVO LEPO DA RADI SVUDA
                                                    "Images (*.png *.jpg)");
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

    if(!qim.isNull()) {
        qim = toGrayscale(qim);

        qim = applyFilter(qim, SOBEL);

        qpm.convertFromImage(qim);

        int image_width  = ui->label4output->width();
        int image_height = ui->label4output->height();

        ui->label4output->setPixmap(qpm.scaled(image_width, image_height, Qt::KeepAspectRatio));
    }
}

void MainWindow::button_median_clicked() {
    QPixmap qpm;
    QImage qim;
    qim = imp.get_image();

    if(!qim.isNull()) {
        qim = applyFilter(qim, MEDIAN);

        qpm.convertFromImage(qim);

        int image_width  = ui->label4output->width();
        int image_height = ui->label4output->height();

        ui->label4output->setPixmap(qpm.scaled(image_width, image_height, Qt::KeepAspectRatio));
    }
}

void MainWindow::button_sharpening_clicked()
{
    QPixmap qpm;
    QImage qim;
    qim = imp.get_image();

    if(!qim.isNull()) {
        qim = applyFilter(qim, SHARPENING);

        qpm.convertFromImage(qim);

        int image_width  = ui->label4output->width();
        int image_height = ui->label4output->height();

        ui->label4output->setPixmap(qpm.scaled(image_width, image_height, Qt::KeepAspectRatio));
    }
}

void MainWindow::button_blur_clicked()
{
    QPixmap qpm;
    QImage qim;
    qim = imp.get_image();

    if(!qim.isNull()) {
        qim = applyFilter(qim, BLUR);

        qpm.convertFromImage(qim);

        int image_width  = ui->label4output->width();
        int image_height = ui->label4output->height();

        ui->label4output->setPixmap(qpm.scaled(image_width, image_height, Qt::KeepAspectRatio));
    }
}

void MainWindow::button_emboss_clicked()
{
    QPixmap qpm;
    QImage qim;
    qim = imp.get_image();

    if(!qim.isNull()) {
        qim = applyFilter(qim, EMBOSS);

        qpm.convertFromImage(qim);

        int image_width  = ui->label4output->width();
        int image_height = ui->label4output->height();

        ui->label4output->setPixmap(qpm.scaled(image_width, image_height, Qt::KeepAspectRatio));
    }

}

void MainWindow::on_button_save_image_clicked()
{
    QString file_name = QFileDialog::getSaveFileName(this, "Save File",
                                                         "~/Pictures/filtered_image.jpg");

    QPixmap const* pix = ui->label4output->pixmap();
    if(pix) {
        QImage image(pix->toImage());
        image.save(file_name);
    }
}


void MainWindow::on_button_filter_compression_clicked()
{
    QPixmap qpm;
    QImage qim;
    qim = imp.get_image();

    if(!qim.isNull()) {
        qim = toGrayscale(qim);

        int img_dim = 28;

        if(qim.width() != img_dim|| qim.height()!=img_dim) {
            qim = qim.scaled(img_dim, img_dim, Qt::KeepAspectRatio);
        }

        cv::Mat img(qim.height(), qim.width(),
                    CV_8UC3, qim.bits(), qim.bytesPerLine());

        //apply compression
        ::google::InitGoogleLogging("compression");
        Compressor compressor("src/prototxt_files/compress_deploy_output_image.prototxt",
                              "trained_models/compress_net_iter_65000.caffemodel",
                              "train_mean.binaryproto");

        std::vector<float> compressed = compressor.compress(img);

        std::ofstream out("compressed.txt");

        for(float a : compressed)
            out << a << " ";

        std::cout << "---------- Saved compressed file to compressed.txt ----------" << std::endl;

        out.close();

        qpm.convertFromImage(qim);

        qpm.convertFromImage(qim);

        int image_width  = ui->label4output->width();
        int image_height = ui->label4output->height();

        ui->label4output->setPixmap(qpm.scaled(image_width, image_height, Qt::KeepAspectRatio));
    }
}

void MainWindow::on_button_filter_gauss_clicked() {
    QPixmap qpm;
    QImage qim;
    qim = imp.get_image();

    if(!qim.isNull()) {
        qim = applyFilter(qim, GAUSS);

        qpm.convertFromImage(qim);

        int image_width  = ui->label4output->width();
        int image_height = ui->label4output->height();

        ui->label4output->setPixmap(qpm.scaled(image_width, image_height, Qt::KeepAspectRatio));
    }
}

///////////////////////////////////////////////////////////////////

QImage &MainWindow::toGrayscale(QImage &qim)
{
    for (int i = 0; i < qim.height(); i++)
    {
        int depth =4;
        uchar* scan = qim.scanLine(i);
        for (int j = 0; j < qim.width(); j++) {
            QRgb* rgbpixel = reinterpret_cast<QRgb*>(scan + j*depth);
            int gray = qGray(*rgbpixel);
            *rgbpixel = QColor(gray, gray, gray, qAlpha(*rgbpixel)).rgba();
        }
    }

    return qim;
}

QImage &MainWindow::applyFilter(QImage &qim, int which) {
    current_filter = which;
    if(current_filter==SOBEL) {
        if(!(qim.isGrayscale()))
        {
            qDebug() << "The QImage I got here is not a grayscale image, so there is no way to do the Sobel!" ;
            return qim;
        }
    }

    QImage qim_copy = qim;

    uchar* scan_previous = qim_copy.scanLine(0);
    uchar* scan_current = qim_copy.scanLine(1);
    uchar* scan_next = qim_copy.scanLine(2);

    QList<QFuture<void> > futures;
    for (int i = 1; i < qim.height() - 1; i++)
    {
        uchar* output_scan_current = qim.scanLine(i);

        //std::function<void((int width, uchar* output_scan_current,
        //                   uchar* scan_previous, uchar* scan_current, uchar* scan_next))> filter_func;

        //calculateLineValue(which, qim.width() - 1, output_scan_current, scan_previous, scan_current, scan_next);
        auto future = QtConcurrent::run(calculateLineValue, qim.width() - 1,
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

///////////////////////////////////////////////////////////////////

void MainWindow::calculateLineValue(int width, uchar* output_scan_current,
                                         uchar* scan_previous, uchar* scan_current, uchar* scan_next)
{
    int depth = 4;

    QVector<QVector<double>> filter_matrix1;
    QVector<QVector<double>> filter_matrix2 = {{0,0,0}, {0,0,0}, {0,0,0}};


    switch(current_filter){
        case SOBEL:
                    qDebug() << "USAO U SOBEL";

                    filter_matrix1   = {{ -1,  -2,  -1},
                                        {  0,   0,   0},
                                        {  1,   2,   1}};

                    filter_matrix2  = {{ -1,  0,  1},
                                       {-2,  0,  2},
                                       {-1,  0,  1}};

                    break;
        case SHARPENING:

                    filter_matrix1 = {{ -1,  -1,  -1},
                                     { -1,   9,  -1},
                                     { -1,  -1,  -1}};

                    break;
        case BLUR:
                    filter_matrix1 =  { { 0,    0.2,  0},
                                        { 0.2,  0.2,  0.2},
                                        { 0,    0.2,  0}};
                    break;
        case GAUSS:

                    filter_matrix1 = {{  1.0/16,  2.0/16,   1.0/16},
                                     {  2.0/16,  4.0/16,   2.0/16},
                                     {  1.0/16,  2.0/16,   1.0/16}};
                    break;
        case EMBOSS:

                    filter_matrix1 = { { -1, -1,  0},
                                       { -1,  0,  1},
                                       { 0,   1,  1}};
                    break;
        case MEDIAN:

                    calculateLineValueMedian(width, output_scan_current,
                                             scan_previous,  scan_current, scan_next);
                    return;
    }

    int tmp;

    for (int j = 1; j < width; ++j) {
        int sum_red = 0, sum_blue = 0, sum_green = 0;

        QRgb* pomocna = (reinterpret_cast<QRgb*>(scan_current  + (j  )*depth));
        auto alpha_of_current = qAlpha(*pomocna);

        if(alpha_of_current == 0)
        {
            //ako je piksel providan onda preskoci racunanje
            QRgb* output_current = reinterpret_cast<QRgb*>(output_scan_current  + (j  )*depth);
            *output_current = QColor(0, 0, 0, 0).rgba();
            continue;
        }
        //qRed da bi dohvatio prvu koordinatu/boju, a sve su iste posto je vec grayscale

        //postavili smo tmp matricu na svoje vrednosti, jos samo da izracunamo
        //sta da stavimo na current position i to je to
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));

            sum_red   += r * (filter_matrix1[0][i+1] + filter_matrix2[0][i+1]);
            sum_blue  += b * (filter_matrix1[0][i+1] + filter_matrix2[0][i+1]);
            sum_green += g * (filter_matrix1[0][i+1] + filter_matrix2[0][i+1]);
        }
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_current + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_current + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_current + (j+i)*depth));

            sum_red   += r * (filter_matrix1[1][i+1] + filter_matrix2[1][i+1]);
            sum_blue  += b * (filter_matrix1[1][i+1] + filter_matrix2[1][i+1]);
            sum_green += g * (filter_matrix1[1][i+1] + filter_matrix2[1][i+1]);
        }
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_next + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_next + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_next + (j+i)*depth));

            sum_red   += r * (filter_matrix1[2][i+1] + filter_matrix2[1][i+1]);
            sum_blue  += b * (filter_matrix1[2][i+1] + filter_matrix2[1][i+1]);
            sum_green += g * (filter_matrix1[2][i+1] + filter_matrix2[1][i+1]);
        }

        // za svaki slucaj, provera
        if(sum_red > 255)
            sum_red = 255;
        if(sum_red < 0)
            sum_red = 0;

        if(sum_blue > 255)
            sum_blue = 255;
        if(sum_blue < 0)
            sum_blue = 0;

        if(sum_green > 255)
            sum_green = 255;
        if(sum_green < 0)
            sum_green = 0;

        QRgb* output_current = reinterpret_cast<QRgb*>(output_scan_current  + (j  )*depth);
        *output_current = QColor(sum_red, sum_green, sum_blue, alpha_of_current).rgba();
    }


}


void MainWindow::calculateLineValueMedian(int width, uchar* output_scan_current, uchar* scan_previous, uchar* scan_current, uchar* scan_next)
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
}


///////////////////////////////////////////////////////////////////

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
