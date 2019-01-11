#include <caffe/caffe.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "net.hpp"

using namespace caffe;
using std::string;

void compression(string model_file, string trained_file, string mean_file, string image) {
	::google::InitGoogleLogging("compression"); 

	Compressor compressor(model_file, trained_file, mean_file); 

	cv::Mat img;
	img = cv::imread(image, -1);

	vector<float> compressed = compressor.compress(img);

	std::ofstream out("compressed.txt");

	for(float a : compressed)
		out << a << " ";

	std::cout << "---------- Saved compressed file to compressed.txt ----------" << std::endl; 

	out.close();
}

int main(int argc, char** argv) { 
	if (argc != 5 && argc != 4) { 
		std::cerr << "Usage: " << argv[0] 
				  << " deploy.prototxt network.caffemodel" 
				  << " mean.binaryproto img.jpg" << std::endl; 
		return 1; 
	} 

	string model_file   = argv[1]; 
	string trained_file = argv[2]; 
	string mean_file    = argv[3]; 
	string image        = argv[4]; 

	compression( model_file,  trained_file,  mean_file,  image);

	return 0;
}
