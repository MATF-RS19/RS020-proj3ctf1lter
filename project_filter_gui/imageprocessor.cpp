#include "imageprocessor.h"

#include <QPixmap>

ImageProcessor::ImageProcessor(QObject *parent) : QObject(parent)
{

}

void ImageProcessor::load_image(QString file_path)
{
    image = QPixmap(file_path).toImage();
}

const QImage& ImageProcessor::get_image() const
{
    return image;
}
