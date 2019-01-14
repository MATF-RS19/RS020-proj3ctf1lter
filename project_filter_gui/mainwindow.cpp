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

#define SOBEL 0
#define MEDIAN 1
#define GAUSS 2
#define EMBOSS 3
#define SHARPENING 4
#define BLUR 5

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
            *rgbpixel = QColor(gray, gray, gray, qAlpha(*rgbpixel)).rgba();
        }
    }

    return qim;
}

QImage &MainWindow::applyFilter(QImage &qim, int which) {

    if(which==SOBEL) {
        if(!(qim.isGrayscale()))
        {
            qDebug() << "The QImage I got here is not a grayscale image, so there is no way I'm doing the Sobel!" ;
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

        std::function<void((int width, uchar* output_scan_current,
                           uchar* scan_previous, uchar* scan_current, uchar* scan_next))> filter_func;

        switch (which) {
        case SOBEL:
            filter_func = MainWindow::calculateLineValueSobel;
            break;
        case MEDIAN:
            filter_func = MainWindow::calculateLineValueMedian;
            break;
        case GAUSS:
            filter_func = MainWindow::calculateLineValueGauss;
            break;
        case EMBOSS:
            filter_func = MainWindow::calculateLineValueEmboss;
            break;
        case SHARPENING:
            filter_func = MainWindow::calculateLineValueSharpening;
            break;
        case BLUR:
            filter_func = MainWindow::calculateLineValueBlur;
            break;
        default:
            break;
        }

        auto future = QtConcurrent::run(filter_func, qim.width() - 1,
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

void MainWindow::calculateLineValueSobel(int width, uchar* output_scan_current,
                                         uchar* scan_previous, uchar* scan_current, uchar* scan_next)
{
    int depth = 4;

    QVector<QVector<int>> sobel_horizontal = {{ -1,  -2,  -1},
                                              {  0,   0,   0},
                                              {  1,   2,   1}};

    QVector<QVector<int>> sobel_vertical   = {{ -1,  0,  1},
                                               {-2,  0,  2},
                                               {-1,  0,  1}};

   //sobel_vertical = {{1,01,01}, {01, 1, 01}, {01, 01, 01}};
   //sobel_horizontal ={{0,0,0}, {0, 0, 0}, {0, 0, 0}} ;

    int tmp;

    for (int j = 1; j < width; ++j) {

        int sum_h = 0;
        int sum_v = 0;

        //qRed da bi dohvatio prvu koordinatu/boju, a sve su iste posto je vec grayscale

        //postavili smo tmp matricu na svoje vrednosti, jos samo da izracunamo
        //sta da stavimo na current position i to je to
        for (int i = -1; i < 2; ++i) {
            tmp = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            sum_h += tmp * sobel_horizontal[0][i+1];
            sum_v += tmp * sobel_vertical[0][i+1];
        }
        for (int i = -1; i < 2; ++i) {
            tmp = qRed(*reinterpret_cast<QRgb*>(scan_current + (j+i)*depth));
            sum_h += tmp * sobel_horizontal[1][i+1];
            sum_v += tmp * sobel_vertical[1][i+1];
        }
        for (int i = -1; i < 2; ++i) {
            tmp = qRed(*reinterpret_cast<QRgb*>(scan_next + (j+i)*depth));
            sum_h += tmp * sobel_horizontal[2][i+1];
            sum_v += tmp * sobel_vertical[2][i+1];
        }

        int sum = sum_h + sum_v;
        // za svaki slucaj, mada ne bi trebalo da upadnemo ovde ikad -- izgleda da upadamo prilicno cesto
        if(sum > 255)
            sum = 255;
        if(sum < 0)
            sum = 0;

        QRgb* pomocna = (reinterpret_cast<QRgb*>(scan_current  + (j  )*depth));
        QRgb* output_current = reinterpret_cast<QRgb*>(output_scan_current  + (j  )*depth);
        *output_current = QColor(sum, sum, sum, qAlpha(*pomocna)).rgba();
    }
}

void MainWindow::calculateLineValueSharpening(int width, uchar* output_scan_current, uchar* scan_previous, uchar* scan_current, uchar* scan_next)
{
    int depth = 4;

    QVector<QVector<int>> sharpening_matrix = {{ -1,  -1,  -1},
                                               { -1,   9,  -1},
                                               { -1,  -1,  -1}};

    for (int j = 1; j < width; ++j) {

        int sum_red = 0, sum_green = 0, sum_blue = 0;

        //qRed da bi dohvatio prvu koordinatu/boju, a sve su iste posto je vec grayscale

        //postavili smo tmp matricu na svoje vrednosti, jos samo da izracunamo
        //sta da stavimo na current position i to je to
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));

            sum_red   += r * sharpening_matrix[0][i+1];
            sum_blue  += b * sharpening_matrix[0][i+1];
            sum_green += g * sharpening_matrix[0][i+1];

        }
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));

            sum_red   += r * sharpening_matrix[1][i+1];
            sum_blue  += b * sharpening_matrix[1][i+1];
            sum_green += g * sharpening_matrix[1][i+1];

        }
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));

            sum_red   += r * sharpening_matrix[2][i+1];
            sum_blue  += b * sharpening_matrix[2][i+1];
            sum_green += g * sharpening_matrix[2][i+1];

        }

        // za svaki slucaj, mada ne bi trebalo da upadnemo ovde ikad -- izgleda da upadamo prilicno cesto
        if(sum_red > 255)
            sum_red = 255;
        if(sum_red < 0)
            sum_red = 0;

        if(sum_blue > 255)
            sum_blue = 255;
        if(sum_blue < 0)
            sum_blue = 0;

        if(sum_blue > 255)
            sum_blue = 255;
        if(sum_blue < 0)
            sum_blue = 0;

        QRgb* pomocna = (reinterpret_cast<QRgb*>(scan_current  + (j  )*depth)); //sluzi da sacuvamo alfa kanal tekuceg piksela, da bude kakav je bio
        QRgb* output_current = reinterpret_cast<QRgb*>(output_scan_current  + (j  )*depth);
        *output_current = QColor(sum_red, sum_green, sum_blue, qAlpha(*pomocna)).rgba();
    }
}

void MainWindow::calculateLineValueBlur(int width, uchar *output_scan_current, uchar *scan_previous, uchar *scan_current, uchar *scan_next)
{
    int depth = 4;


    QVector<QVector<double>> blur_matrix = {        { 0,    0.2,  0},
                                                    { 0.2,  0.2,  0.2},
                                                    { 0,    0.2,  0}};


    for (int j = 1; j < width; ++j) {

        int sum_red = 0, sum_green = 0, sum_blue = 0;

        //qRed da bi dohvatio prvu koordinatu/boju, a sve su iste posto je vec grayscale

        //postavili smo tmp matricu na svoje vrednosti, jos samo da izracunamo
        //sta da stavimo na current position i to je to
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));

            sum_red += r * blur_matrix[0][i+1];
            sum_blue += b * blur_matrix[0][i+1];
            sum_green += g * blur_matrix[0][i+1];

        }
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));

            sum_red += r * blur_matrix[1][i+1];
            sum_blue += b * blur_matrix[1][i+1];
            sum_green += g * blur_matrix[1][i+1];

        }
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));

            sum_red += r * blur_matrix[2][i+1];
            sum_blue += b * blur_matrix[2][i+1];
            sum_green += g * blur_matrix[2][i+1];

        }

        // za svaki slucaj, mada ne bi trebalo da upadnemo ovde ikad -- izgleda da upadamo prilicno cesto
        if(sum_red > 255)
            sum_red = 255;
        if(sum_red < 0)
            sum_red = 0;

        if(sum_blue > 255)
            sum_blue = 255;
        if(sum_blue < 0)
            sum_blue = 0;

        if(sum_blue > 255)
            sum_blue = 255;
        if(sum_blue < 0)
            sum_blue = 0;

        QRgb* pomocna = (reinterpret_cast<QRgb*>(scan_current  + (j  )*depth)); //sluzi da sacuvamo alfa kanal tekuceg piksela, da bude kakav je bio
        QRgb* output_current = reinterpret_cast<QRgb*>(output_scan_current  + (j  )*depth);
        *output_current = QColor(sum_red, sum_green, sum_blue, qAlpha(*pomocna)).rgba();
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

void MainWindow::calculateLineValueGauss(int width, uchar* output_scan_current,
                                         uchar* scan_previous, uchar* scan_current, uchar* scan_next)
{
    int depth = 4;

    QVector<QVector<float>> gauss = {{  1.0/16,  2.0/16,   1.0/16},
                                     {  2.0/16,  4.0/16,   2.0/16},
                                     {  1.0/16,  2.0/16,   1.0/16}};
    QRgb* pixel;

    for (int j = 1; j < width; ++j) {

        int red    = 0,
            green  = 0,
            blue   = 0;

        for (int i = -1; i < 2; ++i) {
            pixel   = reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth);
            red    += (int) (qRed(  *pixel)   * gauss[0][i+1]),
            green  += (int) (qGreen(*pixel)   * gauss[0][i+1]),
            blue   += (int) (qBlue( *pixel)   * gauss[0][i+1]);
        }
        for (int i = -1; i < 2; ++i) {
            pixel   =  reinterpret_cast<QRgb*>(scan_current+ (j+i)*depth);
            red    += (int) (qRed(  *pixel)   * gauss[1][i+1]),
            green  += (int) (qGreen(*pixel)   * gauss[1][i+1]),
            blue   += (int) (qBlue( *pixel)   * gauss[1][i+1]);
        }
        for (int i = -1; i < 2; ++i) {
            pixel   =  reinterpret_cast<QRgb*>(scan_next + (j+i)*depth);
            red    += (int) (qRed(  *pixel)   * gauss[2][i+1]),
            green  += (int) (qGreen(*pixel)   * gauss[2][i+1]),
            blue   += (int) (qBlue( *pixel)   * gauss[2][i+1]);
        }

        QRgb* output_current = reinterpret_cast<QRgb*>(output_scan_current  + (j  )*depth);
        *output_current = QColor(red, green, blue).rgba();
    }
}

void MainWindow::calculateLineValueEmboss(int width, uchar *output_scan_current, uchar *scan_previous, uchar *scan_current, uchar *scan_next)
{
    int depth = 4;

    QVector<QVector<double>> blur_matrix = {        { -1, -1,  0},
                                                    { -1,  0,  1},
                                                    { 0,   1,  1}};

    for (int j = 1; j < width; ++j) {

        int sum_red = 0, sum_green = 0, sum_blue = 0;

        //qRed da bi dohvatio prvu koordinatu/boju, a sve su iste posto je vec grayscale

        //postavili smo tmp matricu na svoje vrednosti, jos samo da izracunamo
        //sta da stavimo na current position i to je to
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));

            sum_red += r * blur_matrix[0][i+1];
            sum_blue += b * blur_matrix[0][i+1];
            sum_green += g * blur_matrix[0][i+1];

        }
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));

            sum_red += r * blur_matrix[1][i+1];
            sum_blue += b * blur_matrix[1][i+1];
            sum_green += g * blur_matrix[1][i+1];

        }
        for (int i = -1; i < 2; ++i) {
            auto r = qRed(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto g = qGreen(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));
            auto b = qBlue(*reinterpret_cast<QRgb*>(scan_previous + (j+i)*depth));

            sum_red += r * blur_matrix[2][i+1];
            sum_blue += b * blur_matrix[2][i+1];
            sum_green += g * blur_matrix[2][i+1];

        }

        // za svaki slucaj, mada ne bi trebalo da upadnemo ovde ikad -- izgleda da upadamo prilicno cesto
        if(sum_red > 255)
            sum_red = 255;
        if(sum_red < 0)
            sum_red = 0;

        if(sum_blue > 255)
            sum_blue = 255;
        if(sum_blue < 0)
            sum_blue = 0;

        if(sum_blue > 255)
            sum_blue = 255;
        if(sum_blue < 0)
            sum_blue = 0;

        QRgb* pomocna = (reinterpret_cast<QRgb*>(scan_current  + (j  )*depth)); //sluzi da sacuvamo alfa kanal tekuceg piksela, da bude kakav je bio
        QRgb* output_current = reinterpret_cast<QRgb*>(output_scan_current  + (j  )*depth);
        *output_current = QColor(sum_red, sum_green, sum_blue, qAlpha(*pomocna)).rgba();
    }

}
