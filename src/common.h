#pragma once

// General modules
#include <sstream>

// SDK modules
#include <ie_core.hpp>
#include <opencv2/opencv.hpp>

// Local modules
#include "person.h"
#include "bd_box.h"

namespace model {
    const std::string DEVICE = "CPU";
    const int16_t END_CODE = -1;
    const uint16_t HUMAN = 1;
    const std::string INPUT_BLOB = "image_tensor";
    const std::string MODEL_XML = "/home/bruno/openvino_models/ir/public/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.xml";
    const std::string MODEL_BIN = "/home/bruno/openvino_models/ir/public/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.bin";
    const std::string OUTPUT_BLOB = "DetectionOutput";
    const std::string INPUT_VID = "/home/bruno/Downloads/tcc/tracker/data/raw/MOT20-01.webm";
};

namespace threshold {
    const float CONF = 0.0;
    const float IOU = 0.25;
    const uint16_t WIDTH = 300;
    const uint16_t HEIGHT = 600;
    const uint16_t DIST = 100;
    const uint32_t STACK = 200;
}

namespace display {
    const uint16_t TXT_OFFSET = 10;
}

namespace common {
    void fillBlob(InferenceEngine::Blob::Ptr input_blob, const cv::Mat& img_data);
    void getBdBox(InferenceEngine::Blob::Ptr output_blob, uint16_t orig_width, uint16_t orig_height, uint32_t frame_num,
                  std::vector<BdBox>& bd_box_vec);
    float IOU(const BdBox& a, const BdBox& b);
    void nonMaxSup(std::vector<BdBox>& bd_box_vec);
    void FIFO(const std::vector<BdBox>& bd_box_vec, std::vector<std::vector<BdBox>>& bd_box_stack);
    uint32_t distance(const BdBox& a, const BdBox& b);
    void findNeighbour(const BdBox& bd_box, const std::vector<BdBox>& bd_box_vec, BdBox& neighbour);
    void assignID(std::vector<BdBox>& bd_box_vec, const std::vector<std::vector<BdBox>>& bd_box_stack, uint32_t frame_num,
                  std::vector<Person>& person_vec);
    void show(const std::vector<Person>& person_vec, cv::Mat& img_data, uint32_t frame_num);
};
