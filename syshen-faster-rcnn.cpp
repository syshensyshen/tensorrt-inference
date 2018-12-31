#include "common.h"
#include "argsParser.h"
#include "buffers.h"

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "syshen-faster-rcnn.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

const int poolingH = 6;
const int poolingW = 6;
const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 7;
const int anchorsScaleCount = 6;
const float nmsThreshold = 0.7f;
const float minScale = 16;
const float spatialScale = 0.0625f;
//const float anchorsRatios[anchorsRatioCount] = { 0.5f, 1.0f, 2.0f };
//const float anchorsScales[anchorsScaleCount] = { 8.0f, 16.0f, 32.0f };
const float anchorsRatios[anchorsRatioCount] = { 0.333f, 0.5f, 0.667f, 1.0f, 1.5f, 2.0f, 3.0f };
const float anchorsScales[anchorsScaleCount] = { 2.0f, 3.0f, 5.0f, 9.0f, 16.0f, 32.0f };

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";

class FRCNNPluginFactory : public nvcaffeparser1::IPluginFactoryV2
{
public:

	FRCNNPluginFactory() {};

	virtual nvinfer1::IPluginV2* createPlugin(const char* layerName, const nvinfer1::Weights* weights,
		int nbWeights, const char* libNamespace) override {
		assert(isPluginV2(layerName));
		if (!strcmp(layerName, "RPROIFused")) {
			assert(mPluginRPROI == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			DimsHW RoiPoolShape = DimsHW(poolingH, poolingW);
			Weights RoiPoolingWeights = Weights{ nvinfer1::DataType::kFLOAT, anchorsRatios, anchorsRatioCount };
			Weights RoiPoolingAnchors = Weights{ nvinfer1::DataType::kFLOAT, anchorsScales, anchorsScaleCount };
			mPluginRPROI = createRPNROIPlugin(featureStride, preNmsTop, nmsMaxOut, nmsThreshold, minScale, spatialScale,
				RoiPoolShape, RoiPoolingWeights, RoiPoolingAnchors);
			mPluginRPROI->setPluginNamespace(libNamespace);
			return mPluginRPROI;
		}
		else {
			assert(0);
			return nullptr;
		}
	}

	// caffe parser plugin implementation
	bool isPluginV2(const char* name) override { return !strcmp(name, "RPROIFused"); }

	void destroyPlugin() {
		mPluginRPROI->destroy();
	}

private:
	//FRCNNPluginFactory() {};
	FRCNNPluginFactory(FRCNNPluginFactory &rhs) {};
	FRCNNPluginFactory & operator=(const FRCNNPluginFactory &) {};
	nvinfer1::IPluginV2* mPluginRPROI;
};

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
	IRuntime* runtimebuilder;
	nvinfer1::ICudaEngine * m_Engine;
	IExecutionContext* context;
	cudaStream_t stream;
};

static int processImg(cv::Mat &img, int inputchannels, float *imgData) {
	cv::Mat float_img;
	img.convertTo(float_img, CV_32F);
	std::vector<cv::Mat> splitchannles = std::vector<cv::Mat>(inputchannels);
	if (3 == inputchannels) {
		if (1 == img.channels())
			cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		cv::Scalar_<float> meanValue = cv::Scalar_<float>(104.0f, 117.0f, 123.0f);
		cv::Mat mean_(img.size(), CV_32FC3, meanValue);
		cv::subtract(float_img, mean_, float_img);
		cv::split(float_img, splitchannles);
	}
	else if (1 == inputchannels) {
		if (3 == img.channels())
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		cv::Scalar_<float> meanValue = cv::Scalar_<float>(104.0f, 0.0f, 0.0f);
		cv::Mat mean_(img.size(), CV_32FC1, meanValue);
		cv::subtract(float_img, mean_, float_img);
		splitchannles.emplace_back(float_img);
	}
	else {
		printf("error inputchannels!!\r\n");
		exit(-1);
	}
	int shift_data = sizeof(float) * img.rows * img.cols;
	for (size_t i = 0; i < inputchannels; i++) {
		memcpy(imgData + i * shift_data, splitchannles[i].data, shift_data);
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

static int parserCaffeModel(const std::string& deployFile,                   // name for caffe prototxt
	const std::string& modelFile,                    // name for model
	const std::vector<std::string>& outputs,         // network outputs
	unsigned int batch_size,                       // batch size - NB must be at least as large as the batch we want to run with)
	nvcaffeparser1::IPluginFactoryV2* pluginFactory, // factory for plugin layers
	IHostMemory** trtModelStream) {
	
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactoryV2(pluginFactory);

	std::cout << "Begin parsing model..." << std::endl;
	const IBlobNameToTensor* blobNameToTensor = parser->parse(
		deployFile.c_str(),
		modelFile.c_str(),
		*network,
		DataType::kFLOAT);
	std::cout << "End parsing model..." << std::endl;
	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(batch_size);
	builder->setMaxWorkspaceSize(1_GB); // we need about 6MB of scratch space for the plugin layer for batch size 5

	std::cout << "Begin building engine..." << std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	std::cout << "End building engine..." << std::endl;

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	(*trtModelStream) = engine->serialize();

	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();

	return 0;
}

int syshen_TesnroRTBuild(frcHandle *handle_t, buildparam *param) {

	syshenRTCaffeCantainer *cantainer = (syshenRTCaffeCantainer *)malloc(sizeof(syshenRTCaffeCantainer));
	FRCNNPluginFactory pluginFactory;
	IHostMemory* trtModelStream{ nullptr };
	parserCaffeModel(param->protostr, param->modelstr, 
		std::vector<std::string>{OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1, OUTPUT_BLOB_NAME2},
		param->batch_size, &pluginFactory, &trtModelStream);
	initLibNvInferPlugins(&gLogger, "");
	IRuntime* runtimebuilder = nvinfer1::createInferRuntime(gLogger);

	if (param->dlaCoreSize >= 0) {
		runtimebuilder->setDLACore(param->dlaCoreSize);
	}
	assert(trtModelStream != nullptr);
	pluginFactory.destroyPlugin();
	ICudaEngine* m_Engine = runtimebuilder->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
	assert(m_Engine != nullptr);
	trtModelStream->destroy();
	IExecutionContext* context = m_Engine->createExecutionContext();
	assert(context != nullptr);	

	/*int inputIndex0 = m_Engine->getBindingIndex(INPUT_BLOB_NAME0),
		inputIndex1 = m_Engine->getBindingIndex(INPUT_BLOB_NAME1),
		outputIndex0 = m_Engine->getBindingIndex(OUTPUT_BLOB_NAME0),
		outputIndex1 = m_Engine->getBindingIndex(OUTPUT_BLOB_NAME1),
		outputIndex2 = m_Engine->getBindingIndex(OUTPUT_BLOB_NAME2);*/

	void *input_data = NULL, *im_info = NULL;
	void *rois = NULL, *bbox_pred = NULL, *cls_prob = NULL;
	int input_size = param->batch_size * param->channels * param->height *param->width * sizeof(float);
	int rois_size = param->batch_size * param->nmsMaxOut * 4 * sizeof(float);
	int bbox_pred_size = param->batch_size * param->nmsMaxOut * param->cls_num * 4 * sizeof(float);
	int cls_prob_size = param->batch_size * param->nmsMaxOut * param->cls_num * sizeof(float);
	
	cantainer->buffers.buffer_size = m_Engine->getNbBindings();
	/*int inputIndex0 = m_Engine->getBindingIndex(INPUT_BLOB_NAME0);
	Dims dims0 = m_Engine->getBindingDimensions(inputIndex0);
	int inputIndex1 = m_Engine->getBindingIndex(INPUT_BLOB_NAME1);
	Dims dims1 = m_Engine->getBindingDimensions(inputIndex1);
	int outputIndex0 = m_Engine->getBindingIndex(OUTPUT_BLOB_NAME0);
	Dims dims2 = m_Engine->getBindingDimensions(outputIndex0);
	int outputIndex1 = m_Engine->getBindingIndex(OUTPUT_BLOB_NAME1);
	Dims dims3 = m_Engine->getBindingDimensions(outputIndex1);
	int outputIndex2 = m_Engine->getBindingIndex(OUTPUT_BLOB_NAME2);
	Dims dims4 = m_Engine->getBindingDimensions(outputIndex2);*/
	cantainer->buffers.buffers = (void **)malloc(sizeof(float) * m_Engine->getNbBindings());
	cudaMalloc(&input_data, input_size);
	cudaMalloc(&im_info, sizeof(float) * 3);
	cudaMalloc(&rois, rois_size);
	cudaMalloc(&bbox_pred, bbox_pred_size);
	cudaMalloc(&cls_prob, cls_prob_size);
	cantainer->buffers.buffers[0] = input_data;
	cantainer->buffers.buffers[1] = im_info;
	cantainer->buffers.buffers[2] = bbox_pred;
	cantainer->buffers.buffers[3] = cls_prob;
	cantainer->buffers.buffers[4] = rois;

	cantainer->context = context;
	cantainer->m_Engine = m_Engine;
	cantainer->runtimebuilder = runtimebuilder;
	cudaStreamCreate(&cantainer->stream);
	*handle_t = cantainer;

	return 0;
}

int syshen_TesnroRTInference(frcHandle handle_t, buildparam *param, std::vector<cv::Mat> &imgs, 
	std::vector<Iresult> &results) {
	syshenRTCaffeCantainer *cantainer = (syshenRTCaffeCantainer *)handle_t;
	std::vector<cv::Mat> resizeImgs;
	syshenRTCaffeCantainer *cantainer = (syshenRTCaffeCantainer *)handle_t;
	for (size_t imgIndex = 0; imgIndex < imgs.size(); imgIndex++) {
		cv::Mat subResizeImg;
		cv::resize(imgs[imgIndex], subResizeImg, cv::Size(param->width, param->height));
		resizeImgs.push_back(subResizeImg);
	}
	std::vector<cv::Mat> resizeImgs;
	int shift_data = param->batch_size * param->channels * param->height * param->width * sizeof(float);
	float *imgData = (float *)malloc(shift_data);
	processImg(resizeImgs, param->channels, imgData);

	int input_size = param->batch_size * param->channels * param->height *param->width * sizeof(float);
	int rois_size = param->batch_size * param->nmsMaxOut * 4 * sizeof(float);
	int bbox_pred_size = param->batch_size * param->nmsMaxOut * param->cls_num * 4 * sizeof(float);
	int cls_prob_size = param->batch_size * param->nmsMaxOut * param->cls_num * sizeof(float);
	float imInfo[60]; // input im_info
	float* rois = (float *)malloc(rois_size);
	float* bboxPreds = (float *)malloc(bbox_pred_size);
	float* clsProbs = (float *)malloc(cls_prob_size);
	cudaMemcpyAsync(cantainer->buffers.buffers[0], imgData, input_size, cudaMemcpyHostToDevice, cantainer->stream);   // data
	cudaMemcpyAsync(cantainer->buffers.buffers[1], imInfo, sizeof(float) * 3, cudaMemcpyHostToDevice, cantainer->stream); // im_info
	cantainer->context->enqueue(param->batch_size, cantainer->buffers.buffers, cantainer->stream, NULL);
	cudaMemcpyAsync(bboxPreds, cantainer->buffers.buffers[2], bbox_pred_size, cudaMemcpyDeviceToHost, cantainer->stream); // bbox_pred
	cudaMemcpyAsync(clsProbs, cantainer->buffers.buffers[3], cls_prob_size, cudaMemcpyDeviceToHost, cantainer->stream);  // cls_prob
	cudaMemcpyAsync(rois, cantainer->buffers.buffers[4], rois_size, cudaMemcpyDeviceToHost, cantainer->stream); // rois
	
	

	free(rois);
	free(bboxPreds);
	free(clsProbs);
	free(imgData);
	return 0;
}

int syshenTensorRTRelease(frcHandle handle_t) {
	syshenRTCaffeCantainer *cantainer = (syshenRTCaffeCantainer *)handle_t;
	//nvinfer1::IBuilder * builder = cantainer->builder;
	//nvinfer1::INetworkDefinition *network = cantainer->network;
	//nvcaffeparser1::ICaffeParser *parser = cantainer->parser;
	IRuntime* runtimebuilder = cantainer->runtimebuilder;
	nvinfer1::ICudaEngine * m_Engine = cantainer->m_Engine;
	IExecutionContext* context = cantainer->context;
	//network->destroy();
	context->destroy();
	m_Engine->destroy();
	runtimebuilder->destroy();
	
	cudaStreamDestroy(cantainer->stream);
	for (size_t i = 0; i < cantainer->buffers.buffer_size; i++) {
		cudaFree(cantainer->buffers.buffers[i]);
	}
	free(cantainer);

	handle_t = NULL;
	return 0;
}
