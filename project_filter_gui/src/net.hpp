#ifndef __NET_HPP
#define __NET_HPP

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using std::string;

class Compressor {
public:
	Compressor(const string& model_file,
		   const string& trained_file,
		   const string& mean_file);

	vector<float> compress(const cv::Mat& img);

private:
	void set_mean_from_file(const string& mean_file);

	//wraps input layer's BLOBs in vector of opencv matrices for convinience
	void wrap_input_layer(vector<cv::Mat>* input_channels);

	void preprocess_and_set_image(const cv::Mat& img,
					vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
};

#endif