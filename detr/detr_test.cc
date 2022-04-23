#include <opencv2/imgcodecs.hpp>

#include "detr.hh"

int main(int argc, char** argv) {
    cv::Mat img = cv::imread("../demo.jpg");

    DETR detr("../detr.trt8");
    std::vector<BBox> bboxes;
    detr.Detect(img, &bboxes);
    return 0;
}
