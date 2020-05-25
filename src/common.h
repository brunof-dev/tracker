#pragma once

// SDK modules
#include <ie_core.hpp>
#include <opencv2/opencv.hpp>

namespace model {
    const std::string DEVICE = "CPU";
    const int16_t END_CODE = -1;
    const uint16_t HUMAN = 1;
    const std::string INPUT_BLOB = "image_tensor";
    const std::string MODEL_XML = "/home/bruno/openvino_models/ir/public/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.xml";
    const std::string MODEL_BIN = "/home/bruno/openvino_models/ir/public/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.bin";
    const std::string OUTPUT_BLOB = "DetectionOutput";
    const std::string INPUT_VID = "/home/bruno/Downloads/tcc/track/data/raw/MOT20-01.webm";
};

namespace threshold {
    const float CONF = 0.0;
    const float IOU = 0.25;
    const uint16_t WIDTH = 300;
    const uint16_t HEIGHT = 600;
}


struct BdBox {
    float conf;
    uint16_t xmin;
    uint16_t ymin;
    uint16_t xmax;
    uint16_t ymax;
};

namespace common {
    void fillBlob(InferenceEngine::Blob::Ptr input_blob, cv::Mat img_data);
    void getBdBox(std::vector<BdBox>& bd_box_vec, InferenceEngine::Blob::Ptr output_blob, uint16_t orig_width,
                  uint16_t orig_height);
    float IOU(BdBox a, BdBox b);
    void nonMaxSup(std::vector<BdBox>& bd_box_vec);
    BdBox findNeighbour(BdBox bd_box, cv::Mat img_data);
};
