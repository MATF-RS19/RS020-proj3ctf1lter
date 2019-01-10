#include <caffe/caffe.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using std::string;

vector<float> entering_image;

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

Compressor::Compressor(const string& model_file,
             const string& trained_file,
             const string& mean_file) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif //CPU_ONLY
 /* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	if(mean_file != "") {
		/* Load the binaryproto mean file. */
		set_mean_from_file(mean_file);
	}
}

void Compressor::set_mean_from_file(const string& mean_file) {

	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	 * filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

vector<float> Compressor::compress(const cv::Mat& img) {

	/* Copy the input layer to a vector and insert our image into it*/
	Blob<float>* input_layer = net_->input_blobs()[0],
				 *output_layer = net_->output_blobs()[0];
	input_layer->Reshape(1, num_channels_,
						 input_geometry_.height, input_geometry_.width);

	/* Forward dimension change to all layers. */
	net_->Reshape();

	vector<cv::Mat> input_channels;
	wrap_input_layer(&input_channels);

	preprocess_and_set_image(img, &input_channels);

	int img_dim = input_layer->width(), size = img_dim*img_dim;

	//save a copy of image vector before compression
	cv::Mat input_img(img_dim, img_dim, CV_32FC1, input_layer->mutable_cpu_data());
	net_->Forward();
	
    std::cout << "---------- Storing decompressed image as rez.png   ----------" << std::endl;
	cv::Mat decompressed_img(img_dim,img_dim, CV_32FC1, output_layer->mutable_cpu_data());
	cv::imwrite("./rez.png", decompressed_img); 

    //float loss = norm(input_img, decompressed_img, cv::NORM_L2);
	float* diff = new float[size];
	caffe_sub(img_dim*img_dim, input_layer->cpu_data(), output_layer->cpu_data(), diff);
	float loss = caffe_cpu_dot(img_dim*img_dim, diff, diff) / 2.0;
	delete[] diff;

    std::cout << "loss is " << loss << std::endl;

	shared_ptr<Blob<float>> compression_layer = net_->blob_by_name("encode3neuron");

	const float* begin = compression_layer->cpu_data(),
		       * end = begin + compression_layer->count();

	vector<float> compression(begin, end);
	return compression;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Compressor::wrap_input_layer(vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Compressor::preprocess_and_set_image(const cv::Mat& img,
                            vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else {
		sample_resized.convertTo(sample_float, CV_32FC1);
	}

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	 * input layer of the network because it is wrapped by the cv::Mat
	 * objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		  == net_->input_blobs()[0]->cpu_data())
		  << "Input channels are not wrapping the input layer of the network.";
}

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
