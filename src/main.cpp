#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ostream>

#define TYPE (0)
#define OUTPUT_STARTING_IMAGE (1)

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

std::vector<float> entering_image;

class Compressor {
public:
	Compressor(const string& model_file,
		   const string& trained_file,
		   const string& mean_file);

	std::vector<float> compress(const cv::Mat& img);

	void decompress();

private:
	void SetMean(const string& mean_file);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
					std::vector<cv::Mat>* input_channels);

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
		SetMean(mean_file);
	}
}

void Compressor::SetMean(const string& mean_file) {

	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
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

std::vector<float> Compressor::compress(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
						 input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];

	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->count();

	std::vector<float> decompression(begin, end);

	if(output_layer->width() > 1 || output_layer->height() > 1) {
		std::cout << "Deploy.prototxt outputs a image of some sort, storing it as rez.png" << std::endl;
		cv::imwrite("./rez.png", cv::Mat(output_layer->width(),output_layer->height(), TYPE, decompression.data())); 
	} else if(output_layer->channels() > 3) {

		std::cout<<(entering_image.size()==decompression.size())<<std::endl;

		//TODO:count loss for image	

		std::cout << "Deploy.prototxt outputs a decompressed image, storing it as rez.png" << std::endl;
		cv::imwrite("./rez.png", cv::Mat((int)sqrt(output_layer->channels()),(int)sqrt(output_layer->channels()), 
					CV_32FC1, decompression.data())); 
	}

	//TODO:extract middle and change this to compression
	return decompression;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Compressor::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
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

void Compressor::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
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

	Blob<float>* input_layer = net_->input_blobs()[0];

    const float* begin = input_layer->cpu_data();
    const float* end = begin + input_layer->count();

    vector<float> entering(begin, end);

    std::cout << "entering size is " <<entering.size() << std::endl;
	entering_image = entering;

    if(input_layer->width() > 1 || input_layer->height() > 1) {
        std::cout << "Saving image as entering.png" << std::endl;
        cv::Mat mat_entering(input_layer->width(),input_layer->height(), CV_32FC1, entering.data());
        cv::imwrite("./entering.png", mat_entering);
    }

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		  == net_->input_blobs()[0]->cpu_data())
		  << "Input channels are not wrapping the input layer of the network.";
}

//TODO: switch from this to compressing and decompressing in compress()
void Compressor::decompress() {

	std::vector<float> compression;
	std::ifstream in("compressed.txt");
	float val;

	in >> val;
	while(!in.eof()) {
		compression.push_back(val);
		in >> val;
	}

	in.close();

	if(compression.data())
		net_->input_blobs()[0]->set_cpu_data( compression.data() );

	net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
	
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->count();

	std::vector<float> decompression(begin, end);
	float* diff = new float[decompression.size()];
	std::cout << decompression.size() << ", " << entering_image.size() << std::endl;
	if(!(decompression.size()==entering_image.size() && decompression.data() && entering_image.data()))
		std::cout<<"ne valja nesto!!!!"<<std::endl;
	caffe_sub(decompression.size(), decompression.data(), entering_image.data(), diff);
	float loss = caffe_cpu_dot(decompression.size(), diff, diff) / 2;
	std::cout<<"loss is "<<loss<<std::endl;

	delete[] diff;

	if(output_layer->width() > 1 || output_layer->height() > 1) {
		std::cout << "Saving image as decompression.png" << std::endl;
		cv::Mat mat_decompression(output_layer->width(),output_layer->height(), TYPE, decompression.data());
		cv::imwrite("./decompression.png", mat_decompression); 
  	} else if(output_layer->channels()>3) {
		std::cout << "Saving image as decompression.png" << std::endl;
		cv::Mat mat_decompression((int)sqrt(output_layer->channels()),(int)sqrt(output_layer->channels()), TYPE, decompression.data());
		cv::imwrite("./decompression.png", mat_decompression); 
	}
}

int main(int argc, char** argv) { 
	//TODO: lose -d option, keep only option for deploy of whole network
	if (argc != 5 && argc != 4) { 
		std::cerr << "Usage: " << argv[0] 
				  << " [-d] deploy.prototxt network.caffemodel" 
				  << " [mean.binaryproto img.jpg]" << std::endl; 
		return 1; 
	} 

	::google::InitGoogleLogging(argv[0]); 

	//TODO:lose this
	if(argc == 5 && argv[1] == std::string("-d")) {

		string model_file   = argv[2]; 
		string trained_file = argv[3]; 
		Compressor compressor(model_file, trained_file, ""); 

		string file = argv[4]; 
		cv::Mat img;

		compressor.decompress();
		return 0;
	} 

	string model_file   = argv[1]; 
	string trained_file = argv[2]; 
	string mean_file    = argv[3]; 
	Compressor compressor(model_file, trained_file, mean_file); 

	string file = argv[4]; 
	cv::Mat img;
	img = cv::imread(file, -1);

	auto compressed = compressor.compress(img);

	std::ofstream out("compressed.txt");

	for(auto a:compressed)
		out << a << " ";

	std::cout << "---------- Saved compressed file to compressed.txt ----------" << std::endl; 

	out.close();

	return 0;
}

#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This test requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
