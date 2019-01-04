#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QGraphicsScene>
#include <qgraphicsview.h>
#include <qdebug.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->button_filter_sobel, SIGNAL(clicked()), this, SLOT(button_sobel_clicked()));

    connect(ui->button_load_image, SIGNAL(clicked()), this, SLOT(button_load_clicked()));
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

}

void MainWindow::set_image()
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
