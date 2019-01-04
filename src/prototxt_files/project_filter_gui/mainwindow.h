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
public Q_SLOTS:
    void button_load_clicked();
    void button_sobel_clicked();


private:
    Ui::MainWindow *ui;
    ImageProcessor imp;

    void set_image();
};

#endif // MAINWINDOW_H
