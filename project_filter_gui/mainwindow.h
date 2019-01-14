#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "imageprocessor.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    QImage& toGrayscale(QImage& qim);
    QImage& applyMedianFilter(QImage &qim);
    QImage& applySharpeningFilter(QImage &qim);
    QImage& applyBlurFilter(QImage &qim);
    QImage& applyEmbossFilter(QImage &qim);
    QImage& applyFilter(QImage &qim, int which);
    QImage& applyGaussianFilter(QImage &qim);

    static void calculateLineValueSobel(int width, uchar* output_scan_current,
                                        uchar* scan_previous, uchar* scan_current, uchar* scan_next);
    static void calculateLineValueMedian(int width, uchar* output_scan_current,
                                        uchar* scan_previous, uchar* scan_current, uchar* scan_next);
    static void calculateLineValueSharpening(int width, uchar* output_scan_current,
                                        uchar* scan_previous, uchar* scan_current, uchar* scan_next);
    static void calculateLineValueBlur(int width, uchar* output_scan_current,
                                        uchar* scan_previous, uchar* scan_current, uchar* scan_next);
    static void calculateLineValueGauss(int width, uchar* output_scan_current,
                                        uchar* scan_previous, uchar* scan_current, uchar* scan_next);
    static void calculateLineValueEmboss(int width, uchar* output_scan_current,
                                        uchar* scan_previous, uchar* scan_current, uchar* scan_next);
    static void calculateLineValue(int width, uchar* output_scan_current,
                                             uchar* scan_previous, uchar* scan_current, uchar* scan_next);

public Q_SLOTS:
    void button_load_clicked();
    void button_sobel_clicked();
    void button_median_clicked();
    void button_sharpening_clicked();
    void button_blur_clicked();
    void button_emboss_clicked();


private slots:
    void on_button_save_image_clicked();

    void on_button_filter_compression_clicked();

    void on_button_filter_gauss_clicked();

private:
    Ui::MainWindow *ui;
    ImageProcessor imp;

    void set_image();
    void set_sobel_image();
};

#endif // MAINWINDOW_H