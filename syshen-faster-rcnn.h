#ifndef __SYSHEN_RENSORRT_INFERENCE_HEADER__
#define __SYSHEN_RENSORRT_INFERENCE_HEADER__

#include <opencv2/core/core.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>

typedef void * frcHandle;

typedef struct buildparam_ buildparam;
typedef struct Iresult_ Iresult;
struct buildparam_ {
	int batch_size, channels, height, width;
	int cls_num, nmsMaxOut;
	int dlaCoreSize;
	std::vector<std::string> inputTensorNames;
	std::vector<std::string> outputTensorNames;
	std::string protostr, modelstr;
};
struct Iresult_ {
	float score;
	int label;
};

int syshen_TesnroRTBuild(frcHandle *handle_t, buildparam *param);

int syshen_TesnroRTInference(frcHandle handle_t, buildparam *param, cv::Mat &img, Iresult &result);

int syshenTensorRTRelease(frcHandle handle_t);

#endif
