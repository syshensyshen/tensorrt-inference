#include "common.h"
#include "argsParser.h"
#include "buffers.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "syshen_tensorrt.h"


typedef struct syshenRTCaffeCantainer_ syshenRTCaffeCantainer;
typedef struct buffer_ buffer;

struct buffer_ {
	int buffer_size;
	void **buffers;
};

static Logger gLogger;

struct syshenRTCaffeCantainer_ {
	//int batch_size;
	//int dlaCore;
	buffer buffers;
	//nvinfer1::INetworkDefinition *network;	
	nvinfer1::ICudaEngine * m_Engine;
	IExecutionContext* context;
	cudaStream_t stream;
};

static int processImg(std::vector<cv::Mat> &imgs, int inputchannels, float *imgData) {
	int shift_data = 0;
	for (size_t index = 0; index < imgs.size(); index++) {
		cv::Mat float_img;
		cv::Mat img = imgs[index];

		std::vector<cv::Mat> splitchannles = std::vector<cv::Mat>(inputchannels);
		if (3 == inputchannels) {
			if (1 == img.channels())
				cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
			img.convertTo(float_img, CV_32F);
			cv::Scalar_<float> meanValue = cv::Scalar_<float>(104.0f, 117.0f, 123.0f);
			cv::Mat mean_(img.size(), CV_32FC3, meanValue);
			cv::subtract(float_img, mean_, float_img);
			cv::split(float_img, splitchannles);
		}
		else if (1 == inputchannels) {
			if (3 == img.channels())
				cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
			img.convertTo(float_img, CV_32F);
			cv::Scalar_<float> meanValue = cv::Scalar_<float>(104.0f, 0.0f, 0.0f);
			cv::Mat mean_(img.size(), CV_32FC1, meanValue);
			cv::subtract(float_img, mean_, float_img);
			splitchannles.emplace_back(float_img);
		}
		else {
			printf("error inputchannels!!\r\n");
			exit(-1);
		}
		shift_data = sizeof(float) * img.rows * img.cols;
		for (size_t i = 0; i < inputchannels; i++) {
			memcpy(imgData, splitchannles[i].data, shift_data);
			imgData += img.rows * img.cols;
		}
	}

	return 0;
}

static void getNetworkDataSize(nvinfer1::ICudaEngine * m_Engine, int index, int *data_size) {
	Dims dims = m_Engine->getBindingDimensions(index);
	//DataType dtype = m_Engine->getBindingDataType(index);
	*data_size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
	switch (t)
	{
	case nvinfer1::DataType::kINT32: return 4;
	case nvinfer1::DataType::kFLOAT: return 4;
	case nvinfer1::DataType::kHALF: return 2;
	case nvinfer1::DataType::kINT8: return 1;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;
}

//static int paserCaffeModel() {
//
//
//
//	return 0;
//}

int syshen_TesnroRTBuild(frcHandle *handle_t, buildparam *param) {

	syshenRTCaffeCantainer *cantainer = (syshenRTCaffeCantainer *)malloc(sizeof(syshenRTCaffeCantainer));
	nvinfer1::IBuilder * builder = nvinfer1::createInferBuilder(gLogger);
	nvinfer1::INetworkDefinition *network = builder->createNetwork();
	nvcaffeparser1::ICaffeParser *parser = nvcaffeparser1::createCaffeParser();

	int max_batch_size = builder->getMaxBatchSize();
	//param->batch_size = param->batch_size > max_batch_size ? max_batch_size : param->batch_size;

	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
		param->protostr.c_str(),
		param->modelstr.c_str(),
		*network,
		//nvinfer1::DataType::kHALF);
		nvinfer1::DataType::kFLOAT);

	/*for (auto& s : param->outputTensorNames)
		network->markOutput(*blobNameToTensor->find(s.c_str()));*/

	builder->setMaxBatchSize(param->batch_size);
	//max_batch_size = builder->getMaxBatchSize();
	builder->setMaxWorkspaceSize(16_KB);
	//std::cout << builder->getNbDLACores() << std::endl;
	//cantainer->dlaCore = builder->getNbDLACores() > param->dlaCoreSize ? param->dlaCoreSize : builder->getNbDLACores();
	if (param->dlaCoreSize >= 0) {
		builder->allowGPUFallback(true);
		builder->setFp16Mode(true);
		builder->setDefaultDeviceType(DeviceType::kDLA);
		//builder->setDefaultDeviceType(DeviceType::kGPU);
		builder->setDLACore(param->dlaCoreSize);
	}

	nvinfer1::ICudaEngine * m_Engine = builder->buildCudaEngine(*network);

	int inputIndex = m_Engine->getBindingIndex(param->inputTensorNames[0].c_str());
	int input_size = param->batch_size * param->channels * param->height *param->width;
	//getNetworkDataSize(m_Engine, inputIndex, &input_size);
	void *input_data = NULL, *output_data = NULL;
	int outputIndex = 0;
	//m_Engine->getBindingIndex(param->outputTensorNames[0].c_str());
	int output_size = 0;
	getNetworkDataSize(m_Engine, outputIndex, &output_size);
	cantainer->buffers.buffer_size = m_Engine->getNbBindings();
	cantainer->buffers.buffers = (void **)malloc(sizeof(float) * m_Engine->getNbBindings());
	cudaMalloc(&input_data, input_size * sizeof(float));
	cudaMalloc(&output_data, output_size * sizeof(float) *param->batch_size);
	cantainer->buffers.buffers[0] = input_data;
	cantainer->buffers.buffers[1] = output_data;

	IExecutionContext* context = m_Engine->createExecutionContext();

	builder->destroy();
	network->destroy();
	parser->destroy();
	nvcaffeparser1::shutdownProtobufLibrary();

	//cantainer->builder = builder;
	//cantainer->parser = parser;
	//cantainer->network = network;	
	cantainer->m_Engine = m_Engine;
	cantainer->context = context;
	cudaStreamCreate(&cantainer->stream);
	*handle_t = cantainer;

	return 0;
}

int syshen_TesnroRTInference(frcHandle handle_t, buildparam *param, std::vector<cv::Mat> &imgs,
	std::vector<Iresult> &results) {

	syshenRTCaffeCantainer *cantainer = (syshenRTCaffeCantainer *)handle_t;
	//cv::Mat InferenceImg;
	//std::vector<cv::Mat> splitchannles;
	std::vector<cv::Mat> resizeImgs;
	int inputIndex = cantainer->m_Engine->getBindingIndex(param->inputTensorNames[0].c_str());
	//Dims dimsIn = cantainer->m_Engine->getBindingDimensions(inputIndex);
	for (size_t imgIndex = 0; imgIndex < imgs.size(); imgIndex++) {
		cv::Mat subResizeImg;
		cv::resize(imgs[imgIndex], subResizeImg, cv::Size(param->width, param->height));
		resizeImgs.push_back(subResizeImg);
	}

	int shift_data = param->batch_size * param->channels * param->height * param->width * sizeof(float);
	float *imgData = (float *)malloc(shift_data);
	//int shift_data = imgfs * sizeof(float);
	processImg(resizeImgs, param->channels, imgData);
	cudaError stat = cudaMemcpyAsync(cantainer->buffers.buffers[inputIndex], imgData, shift_data,
		cudaMemcpyHostToDevice, cantainer->stream);
	//std::cout << stat << std::endl;
	cantainer->context->execute(param->batch_size, &cantainer->buffers.buffers[0]);
	int outputputIndex = cantainer->m_Engine->getBindingIndex(param->outputTensorNames[0].c_str());
	Dims dimsOut = cantainer->m_Engine->getBindingDimensions(outputputIndex);
	int outSize = std::accumulate(dimsOut.d, dimsOut.d + dimsOut.nbDims, 1, std::multiplies<int64_t>());
	float *output = (float *)malloc(param->batch_size * outSize * sizeof(float));
	cudaMemcpyAsync(output, cantainer->buffers.buffers[1], param->batch_size * outSize * sizeof(float),
		cudaMemcpyDeviceToHost, cantainer->stream);
	for (size_t i = 0; i < param->batch_size; i++) {
		for (size_t sub = 0; sub < outSize; sub++) {
			std::cout << output[i * outSize + sub] << " ";
		}
		std::cout << std::endl;
	}
	free(imgData);
	return 0;
}

int syshenTensorRTRelease(frcHandle handle_t) {
	syshenRTCaffeCantainer *cantainer = (syshenRTCaffeCantainer *)handle_t;
	//nvinfer1::IBuilder * builder = cantainer->builder;
	//nvcaffeparser1::ICaffeParser *parser = cantainer->parser;
	//nvinfer1::INetworkDefinition *network = cantainer->network;	
	nvinfer1::ICudaEngine * m_Engine = cantainer->m_Engine;
	IExecutionContext* context = cantainer->context;
	//network->destroy();
	context->destroy();
	m_Engine->destroy();
	//builder->destroy();
	//parser->destroy();
	//nvcaffeparser1::shutdownProtobufLibrary();
	cudaStreamDestroy(cantainer->stream);
	for (size_t i = 0; i < cantainer->buffers.buffer_size; i++) {
		cudaFree(cantainer->buffers.buffers[i]);
	}
	free(cantainer);

	handle_t = NULL;
	return 0;
}
