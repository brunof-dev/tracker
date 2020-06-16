#pragma once

namespace model {
    const std::string DEVICE = "CPU";
    const int16_t END_CODE = -1;
    const uint16_t HUMAN = 1;
    const std::string INPUT_BLOB = "image_tensor";
    const std::string INPUT_VID = "/home/bruno/Downloads/tcc/tracker/data/raw/MOT17-09.webm";
    const std::string MODEL_XML = "/home/bruno/openvino_models/ir/public/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.xml";
    const std::string MODEL_BIN = "/home/bruno/openvino_models/ir/public/ssd_mobilenet_v1_coco/FP32/ssd_mobilenet_v1_coco.bin";
    const std::string OUTPUT_BLOB = "DetectionOutput";
    const std::string OUTPUT_VID = "/home/bruno/Downloads/tcc/tracker/data/detect/MOT17-09.avi";
};

