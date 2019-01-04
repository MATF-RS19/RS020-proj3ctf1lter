#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <QImage>
#include <QObject>

class ImageProcessor : public QObject
{
    Q_OBJECT
public:
    explicit ImageProcessor(QObject *parent = nullptr);

    void load_image(QString file_path);
    const QImage& get_image() const;

signals:

public slots:

private:

    QImage image;
};

#endif // IMAGEPROCESSOR_H
