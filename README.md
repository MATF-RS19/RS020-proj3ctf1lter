# RS020-proj3ctf1lter

Convolutional neural network that compresses images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine running Ubuntu 18.04 for development and testing purposes.

### Prerequisites

What things you need to install the software

```
cmake (version >= 3.10)
caffe (located at ~)
```


### Installing

#### Clone

- Clone this repo to your local machine using 
`git clone https://github.com/MATF-RS19/RS020-proj3ctf1lter`

#### Setup

- Build project using Cmake:

```shell
$ cd RS020-proj3ctf1lter
$ mkdir build && cd build
$ cmake ..
$ make
```

Note: If you want to train CNN on CPU you have to add flag -DCPU_ONLY in `CMakeLists.txt`

### Training

Image used to train CNN needs to be in train folder and train.txt must contain list of all training data (same thing for val).

- First create lmdb from training data:

```shell
$ cd src
$ ./images_to_lmdb.sh
```

- Create image mean for training and val:

```shell
$ cd prototxt_files
$ ~/caffe/build/tools/compute_image_mean ../../build/train_lmdb/ train_mean.binaryproto
$ mv train_mean.binaryproto ../../build/train_mean.binaryproto
$ ~/caffe/build/tools/compute_image_mean ../../build/val_lmdb/ val_mean.binaryproto
$ mv val_mean.binaryproto ../../build/val_mean.binaryproto
```

- Then train neural network:

```shell
$ ~/caffe/build/tools/caffe train -solver compress_solver.prototxt -gpu all
```
Note: Remove -gpu all flag to train on CPU

## Built With

* [caffe](https://github.com/BVLC/caffe) - Fast open framework for deep learning
* [Cmake](https://cmake.org/) - Build tool


## Team

| <a href="https://github.com/v1rTu0Zz" target="_blank">**Nikola Mandic**</a> | <a href="https://github.com/laleee" target="_blank">**Lazar Jovanovic**</a> | <a href="https://github.com/stral0" target="_blank">**Strahinja Mitric**</a> |
| :---: |:---:| :---:|
| [![Nikola](https://avatars1.githubusercontent.com/u/30957582?s=200&v=3)](https://github.com/v1rTu0Zz)    | [![Lazar](https://avatars3.githubusercontent.com/u/15856722?s=200&v=3)](https://github.com/laleee) | [![Strahinja](https://avatars1.githubusercontent.com/u/18012692?s=200&v=3)](https://github.com/stral0)  |
| <a href="https://github.com/v1rTu0Zz" target="_blank">`v1rTu0Zz`</a> | <a href="https://github.com/laleee" target="_blank">`laleee`</a> | <a href="https://github.com/stral0" target="_blank">`stral0`</a> |

See also the list of [contributors](https://github.com/MATF-RS19/RS020-proj3ctf1lter/graphs/contributors) who participated in this project.

