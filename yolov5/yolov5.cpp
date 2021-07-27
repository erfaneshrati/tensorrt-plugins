#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define USE_FP16 // comment out this if want to use FP32
#define DEVICE 0 // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define KEEP_TOPK 100

#define NET s // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
const char *INPUT_NAME = "images";
const char *OUTPUT_COUNTS = "count";
const char *OUTPUT_BOXES = "predicted_det_bboxes";
const char *OUTPUT_SCORES = "predicted_det_scores";
const char *OUTPUT_CLASSES = "predicted_det_labels";

static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max(r, 1);
}

// Creat the engine using only the API and not any parser.
ICudaEngine *createEngine_s(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)
{
    float gd = 0.33;
    float gw = 0.50;
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_NAME, dt, Dims3{ INPUT_H, INPUT_W, 3 });
    assert(data);
    // NHWC to NCHW
    IShuffleLayer* shuffle = network->addShuffle(*data);
    shuffle->setSecondTranspose(Permutation{2, 0, 1});

    // Normalize.
    float _scale = 1. / 255;
    float _shift = 0.f;
    float _power = 1.f;
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, &_scale, 1};
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, &_shift, 1};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, &_power, 1};

    auto scale_layer = network->addScale(*shuffle->getOutput(0), nvinfer1::ScaleMode::kUNIFORM, shift, scale, power);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov5s.wts");

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *scale_layer->getOutput(0), 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);

    auto nms = addBatchedNMSLayer(network, yolo, Yolo::CLASS_NUM, Yolo::MAX_OUTPUT_BBOX_COUNT, KEEP_TOPK, CONF_THRESH, NMS_THRESH);

//    nms->getOutput(0)->setName(OUTPUT_COUNTS);
//    network->markOutput(*nms->getOutput(0));

    nms->getOutput(1)->setName(OUTPUT_BOXES);
    network->markOutput(*nms->getOutput(1));

    IShuffleLayer* shuffle_scores = network->addShuffle(*nms->getOutput(2));
    shuffle_scores->setReshapeDimensions(Dims2{KEEP_TOPK, 1});
    shuffle_scores->getOutput(0)->setName(OUTPUT_SCORES);
    network->markOutput(*shuffle_scores->getOutput(0));

    IShuffleLayer* shuffle_classes = network->addShuffle(*nms->getOutput(3));
    shuffle_classes->setReshapeDimensions(Dims2{KEEP_TOPK, 1});
    shuffle_classes->getOutput(0)->setName(OUTPUT_CLASSES);
    network->markOutput(*shuffle_classes->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }
    return engine;
}

ICudaEngine *createEngine_m(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)
{
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov5m.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------ yolov5 backbone------
    auto focus0 = focus(network, weightMap, *data, 3, 48, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 96, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 96, 96, 2, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 192, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 192, 192, 6, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 384, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 384, 384, 6, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 768, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 768, 768, 5, 9, 13, "model.8");
    // ------ yolov5 head ------
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 768, 768, 2, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 384, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * 384 * 2 * 2));
    for (int i = 0; i < 384 * 2 * 2; i++)
    {
        deval[i] = 1.0;
    }
    Weights deconvwts11{DataType::kFLOAT, deval, 384 * 2 * 2};
    IDeconvolutionLayer *deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 384, DimsHW{2, 2}, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{2, 2});
    deconv11->setNbGroups(384);
    weightMap["deconv11"] = deconvwts11;
    ITensor *inputTensors12[] = {deconv11->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);

    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 768, 384, 2, false, 1, 0.5, "model.13");

    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 192, 1, 1, 1, "model.14");

    Weights deconvwts15{DataType::kFLOAT, deval, 192 * 2 * 2};
    IDeconvolutionLayer *deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 192, DimsHW{2, 2}, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{2, 2});
    deconv15->setNbGroups(192);

    ITensor *inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 384, 192, 2, false, 1, 0.5, "model.17");

    //yolo layer 0
    IConvolutionLayer *det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 192, 3, 2, 1, "model.18");
    ITensor *inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 384, 384, 2, false, 1, 0.5, "model.20");

    //yolo layer 1
    IConvolutionLayer *det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 384, 3, 2, 1, "model.21");
    ITensor *inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 768, 768, 2, false, 1, 0.5, "model.23");
    // yolo layer 2
    IConvolutionLayer *det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    auto nms = addBatchedNMSLayer(network, yolo, Yolo::CLASS_NUM, Yolo::MAX_OUTPUT_BBOX_COUNT, KEEP_TOPK, CONF_THRESH, NMS_THRESH);

    nms->getOutput(0)->setName(OUTPUT_COUNTS);
    network->markOutput(*nms->getOutput(0));

    nms->getOutput(1)->setName(OUTPUT_BOXES);
    network->markOutput(*nms->getOutput(1));

    nms->getOutput(2)->setName(OUTPUT_SCORES);
    network->markOutput(*nms->getOutput(2));

    nms->getOutput(3)->setName(OUTPUT_CLASSES);
    network->markOutput(*nms->getOutput(3));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }

    return engine;
}

ICudaEngine *createEngine_l(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)
{
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov5l.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------ yolov5 backbone------
    auto focus0 = focus(network, weightMap, *data, 3, 64, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 128, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 256, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 256, 256, 9, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 512, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 512, 512, 9, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 1024, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 1024, 1024, 5, 9, 13, "model.8");

    // ------ yolov5 head ------
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 1024, 1024, 3, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 512, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * 512 * 2 * 2));
    for (int i = 0; i < 512 * 2 * 2; i++)
    {
        deval[i] = 1.0;
    }
    Weights deconvwts11{DataType::kFLOAT, deval, 512 * 2 * 2};
    IDeconvolutionLayer *deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 512, DimsHW{2, 2}, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{2, 2});
    deconv11->setNbGroups(512);
    weightMap["deconv11"] = deconvwts11;

    ITensor *inputTensors12[] = {deconv11->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 1024, 512, 3, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 256, 1, 1, 1, "model.14");

    Weights deconvwts15{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer *deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 256, DimsHW{2, 2}, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{2, 2});
    deconv15->setNbGroups(256);
    ITensor *inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 512, 256, 3, false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer *det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 256, 3, 2, 1, "model.18");
    ITensor *inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 512, 512, 3, false, 1, 0.5, "model.20");
    //yolo layer 1
    IConvolutionLayer *det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 512, 3, 2, 1, "model.21");
    ITensor *inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 1024, 1024, 3, false, 1, 0.5, "model.23");
    IConvolutionLayer *det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    auto nms = addBatchedNMSLayer(network, yolo, Yolo::CLASS_NUM, Yolo::MAX_OUTPUT_BBOX_COUNT, KEEP_TOPK, CONF_THRESH, NMS_THRESH);

    nms->getOutput(0)->setName(OUTPUT_COUNTS);
    network->markOutput(*nms->getOutput(0));

    nms->getOutput(1)->setName(OUTPUT_BOXES);
    network->markOutput(*nms->getOutput(1));

    nms->getOutput(2)->setName(OUTPUT_SCORES);
    network->markOutput(*nms->getOutput(2));

    nms->getOutput(3)->setName(OUTPUT_CLASSES);
    network->markOutput(*nms->getOutput(3));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }

    return engine;
}

ICudaEngine *createEngine_x(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)
{
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov5x.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------ yolov5 backbone------
    auto focus0 = focus(network, weightMap, *data, 3, 80, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 160, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 160, 160, 4, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 320, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 320, 320, 12, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 640, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 640, 640, 12, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 1280, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 1280, 1280, 5, 9, 13, "model.8");

    // ------- yolov5 head -------
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 1280, 1280, 4, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 640, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * 640 * 2 * 2));
    for (int i = 0; i < 640 * 2 * 2; i++)
    {
        deval[i] = 1.0;
    }
    Weights deconvwts11{DataType::kFLOAT, deval, 640 * 2 * 2};
    IDeconvolutionLayer *deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 640, DimsHW{2, 2}, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{2, 2});
    deconv11->setNbGroups(640);
    weightMap["deconv11"] = deconvwts11;

    ITensor *inputTensors12[] = {deconv11->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);

    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 1280, 640, 4, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 320, 1, 1, 1, "model.14");

    Weights deconvwts15{DataType::kFLOAT, deval, 320 * 2 * 2};
    IDeconvolutionLayer *deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 320, DimsHW{2, 2}, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{2, 2});
    deconv15->setNbGroups(320);
    ITensor *inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 640, 320, 4, false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer *det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 320, 3, 2, 1, "model.18");
    ITensor *inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 640, 640, 4, false, 1, 0.5, "model.20");
    // yolo layer 1
    IConvolutionLayer *det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 640, 3, 2, 1, "model.21");
    ITensor *inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 1280, 1280, 4, false, 1, 0.5, "model.23");
    // yolo layer 2
    IConvolutionLayer *det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    auto nms = addBatchedNMSLayer(network, yolo, Yolo::CLASS_NUM, Yolo::MAX_OUTPUT_BBOX_COUNT, KEEP_TOPK, CONF_THRESH, NMS_THRESH);

    nms->getOutput(0)->setName(OUTPUT_COUNTS);
    network->markOutput(*nms->getOutput(0));

    nms->getOutput(1)->setName(OUTPUT_BOXES);
    network->markOutput(*nms->getOutput(1));

    nms->getOutput(2)->setName(OUTPUT_SCORES);
    network->markOutput(*nms->getOutput(2));

    nms->getOutput(3)->setName(OUTPUT_CLASSES);
    network->markOutput(*nms->getOutput(3));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream)
{
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = (CREATENET(NET))(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext &context, ICudaEngine *engine, float *input, int *counts, float *boxes, float *scores, float *classes, int batchSize)
{
    // const ICudaEngine &engine = context.getEngine();
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    assert(engine->getNbBindings() == 5);
    void *buffers[5];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_NAME);
    const int countIndex = engine->getBindingIndex(OUTPUT_COUNTS);
    const int bboxIndex = engine->getBindingIndex(OUTPUT_BOXES);
    const int scoreIndex = engine->getBindingIndex(OUTPUT_SCORES);
    const int classIndex = engine->getBindingIndex(OUTPUT_CLASSES);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[countIndex], batchSize * sizeof(int)));
    CHECK(cudaMalloc(&buffers[bboxIndex], batchSize * KEEP_TOPK * 4 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[scoreIndex], batchSize * KEEP_TOPK * sizeof(float)));
    CHECK(cudaMalloc(&buffers[classIndex], batchSize * KEEP_TOPK * sizeof(float)));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(counts, buffers[countIndex], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(boxes, buffers[bboxIndex], batchSize * KEEP_TOPK * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(scores, buffers[scoreIndex], batchSize * KEEP_TOPK * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(classes, buffers[classIndex], batchSize * KEEP_TOPK * sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[countIndex]));
    CHECK(cudaFree(buffers[bboxIndex]));
    CHECK(cudaFree(buffers[scoreIndex]));
    CHECK(cudaFree(buffers[classIndex]));
}

int main(int argc, char **argv)
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    std::string engine_name = STR2(NET);
    engine_name = "yolov5" + engine_name + ".engine";

    initLibNvInferPlugins(&gLogger, "");
    if (argc == 2 && std::string(argv[1]) == "-s")
    {
        IHostMemory *modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else if (argc == 3 && std::string(argv[1]) == "-d")
    {
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0)
    {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    return 0;
}
