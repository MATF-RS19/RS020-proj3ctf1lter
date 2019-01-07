#! /bin/sh
MODEL="../src/prototxt_files/compress_net_snap_iter_100.caffemodel"
IMAGE="../train/George_W_Bush_0410.jpg"

~/caffe/build/tools/caffe train -solver compress_solver.prototxt
cd ../../build/
cmake ..
make
./main ../src/prototxt_files/compress_deploy.prototxt $MODEL train_gray_mean.binaryproto $IMAGE
./main -d ../src/prototxt_files/decompress_deploy.prototxt $MODEL
eog decompression.bmp
