
#include "syshen_tensorrt.h"
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
	///////////////////// param
	std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
	std::vector<std::string> inputTensorNames;
	std::vector<std::string> outputTensorNames;
	std::string protostr = "../../../models/test.prototxt";
	std::string modelstr = "../../../models/test.caffemodel";
	frcHandle handle_t;
	buildparam param;
	param.batch_size = 1;
	param.protostr = protostr;
	param.modelstr = modelstr;
	param.dlaCoreSize = -1;
	param.inputTensorNames.emplace_back("data");
	param.outputTensorNames.emplace_back("prob");
	syshen_TesnroRTBuild(&handle_t, &param);

	cv::Mat img = cv::imread("test.jpg");
	Iresult result;
	syshen_TesnroRTInference(handle_t, &param, img, result);
	return 0;
}