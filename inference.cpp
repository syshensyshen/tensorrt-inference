
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
	param.batch_size = 4;
	param.channels = 3;
	param.height = 192;
	param.width = 240;
	param.protostr = protostr;
	param.modelstr = modelstr;
	param.dlaCoreSize = -1;
	param.inputTensorNames.emplace_back("data");
	param.outputTensorNames.emplace_back("prob");
	syshen_TesnroRTBuild(&handle_t, &param);

	cv::Mat img = cv::imread("./img/5.jpg");
	std::vector<cv::Mat> imgs;
	imgs.push_back(img.clone());
	imgs.push_back(img.clone());
	imgs.push_back(img.clone());
	imgs.push_back(img.clone());
	std::vector<Iresult> results;
	syshen_TesnroRTInference(handle_t, &param, imgs, results);
	syshenTensorRTRelease(handle_t);

	return 0;
}