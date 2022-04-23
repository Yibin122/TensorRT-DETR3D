#include "detr.hh"

#include <fstream>
#include <memory>
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define INPUT_H 800
#define INPUT_W 1199
#define NUM_QUERY 100
#define SCORE_THRESH 0.8f

DETR::DETR(const std::string& engine_path) {
    buffer_size_[0] = 3 * INPUT_H * INPUT_W;
    buffer_size_[1] = NUM_QUERY * 5;
    buffer_size_[2] = NUM_QUERY;
    cudaMalloc(&buffers_[0], buffer_size_[0] * sizeof(float));
    cudaMalloc(&buffers_[1], buffer_size_[1] * sizeof(float));
    cudaMalloc(&buffers_[2], buffer_size_[2] * sizeof(float));
    image_data_.resize(buffer_size_[0]);
    det_bboxes_.resize(buffer_size_[1]);
    det_labels_.resize(buffer_size_[2]);
    cudaStreamCreate(&stream_);
    LoadEngine(engine_path);
}

DETR::~DETR() {
    cudaStreamDestroy(stream_);
    for (auto& buffer : buffers_) {
        cudaFree(buffer);
    }
    if (context_ != nullptr) {
        context_->destroy();
        engine_->destroy();
    }
}

float Clamp(float x) {
    float x_clamp = x;
    // torch.clamp(x, 0, 1)
    if (x_clamp > 1.0f) {
        x_clamp = 1.0f;
    } else if (x_clamp < 0.0f) {
        x_clamp = 0.0f;
    }
    return x_clamp;
}

void DETR::Detect(const cv::Mat& raw_img, std::vector<BBox>* bboxes) {
    // Preprocessing
    cv::Mat img_resize;
    cv::resize(raw_img, img_resize, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);
    // img_resize.convertTo(img_resize, CV_32FC3, 1.0f);
    float mean[3] {123.675f, 116.280f, 103.530f};
    float std[3] = {58.395f, 57.120f, 57.375f};
    uint8_t* data_hwc = reinterpret_cast<uint8_t*>(img_resize.data);
    float* data_chw = image_data_.data();
    for (int c = 0; c < 3; ++c) {
        for (unsigned j = 0, img_size = INPUT_H * INPUT_W; j < img_size; ++j) {
            data_chw[c * img_size + j] = (data_hwc[j * 3 + 2 - c] - mean[c]) / std[c];  //bgr2rgb
        }
    }

    // Do inference
    cudaMemcpyAsync(buffers_[0], image_data_.data(), buffer_size_[0] * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->executeV2(&buffers_[0]);
    cudaMemcpyAsync(det_bboxes_.data(), buffers_[1], buffer_size_[1] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(det_labels_.data(), buffers_[2], buffer_size_[2] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    // cudaStreamSynchronize(stream_);
    
    // PostProcessing
    for (int i = 0; i < NUM_QUERY; ++i) {
        if (det_bboxes_[5 * i + 4] < SCORE_THRESH) {
            continue;
        }
        BBox bbox;
        bbox.x1 = static_cast<float>(INPUT_W) * Clamp(det_bboxes_[5 * i]);
        bbox.y1 = static_cast<float>(INPUT_H) * Clamp(det_bboxes_[5 * i + 1]);
        bbox.x2 = static_cast<float>(INPUT_W) * Clamp(det_bboxes_[5 * i + 2]);
        bbox.y2 = static_cast<float>(INPUT_H) * Clamp(det_bboxes_[5 * i + 3]);
        bbox.score = det_bboxes_[5 * i + 4];
        bbox.class_id = static_cast<int>(det_labels_[i]);
        bboxes->push_back(bbox);
        // TODO: show class name
        cv::rectangle(img_resize, cv::Point2f(bbox.x1, bbox.y1), cv::Point2f(bbox.x2, bbox.y2), cv::Scalar(241, 101, 72), 2);
    }
    cv::imshow("DETR_TRT", img_resize);
    cv::waitKey(0);
}

void DETR::LoadEngine(const std::string& engine_path) {
    std::ifstream in_file(engine_path, std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return;
    }
    in_file.seekg(0, in_file.end);
    int length = in_file.tellg();
    in_file.seekg(0, in_file.beg);
    std::unique_ptr<char[]> trt_model_stream(new char[length]);
    in_file.read(trt_model_stream.get(), length);
    in_file.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_);
    assert(runtime != nullptr);
    engine_ = runtime->deserializeCudaEngine(trt_model_stream.get(), length, nullptr);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);

    runtime->destroy();
}
