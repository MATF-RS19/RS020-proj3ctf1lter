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
    QImage& applySobelFilter(QImage& qim);
    static int calculatePixelValueSobel(int j, uchar* output_scan_current,
                                        uchar* scan_previous, uchar* scan_current, uchar* scan_next);


public Q_SLOTS:
    void button_load_clicked();
    void button_sobel_clicked();


private:
    Ui::MainWindow *ui;
    ImageProcessor imp;

    void set_image();
    void set_sobel_image();
};

#endif // MAINWINDOW_H
