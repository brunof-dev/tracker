#pragma once

// General modules
#include <sstream>
#include <cstdlib>

// SDK modules
#include <ie_core.hpp>
#include <opencv2/opencv.hpp>

// Local modules
#include "person.h"
#include "bd_box.h"
#include "display.h"
#include "model.h"
#include "threshold.h"

namespace common {
    void fillBlob(InferenceEngine::Blob::Ptr input_blob, const cv::Mat& img_data);
    void getBdBox(InferenceEngine::Blob::Ptr output_blob, uint16_t orig_width, uint16_t orig_height, uint32_t frame_num,
                  std::vector<BdBox>& bd_box_vec);
    float IOU(const BdBox& a, const BdBox& b);
    void nonMaxSup(std::vector<BdBox>& bd_box_vec);
    void getHOG(std::vector<BdBox>& bd_box_vec, cv::Mat img_data, const cv::HOGDescriptor& hog_handler);
    void getORB(std::vector<BdBox>& bd_box_vec, cv::Mat img_data, cv::Ptr<cv::ORB> orb_handler);
    uint32_t distance(const BdBox& a, const BdBox& b);
    float distance(const std::vector<float>& a, const std::vector<float>& b);
    float distance(const cv::Mat& a, const cv::Mat& b);
    void findNeighbour(const BdBox& bd_box, const std::vector<BdBox>& bd_box_vec, BdBox& neighbour);
    void findNbVec(const BdBox& bd_box, const std::vector<BdBox>& bd_box_vec, std::vector<BdBox>& nb_vec);
    void matchNbVec(const BdBox& bd_box, const std::vector<BdBox>& nb_vec, BdBox& neighbour);
    void assignID(std::vector<BdBox>& bd_box_vec, const std::vector<std::vector<BdBox>>& bd_box_stack, uint32_t frame_num,
                  std::vector<Person>& person_vec);
    void FIFO(const std::vector<BdBox>& bd_box_vec, std::vector<std::vector<BdBox>>& bd_box_stack);
    void show(const std::vector<Person>& person_vec, cv::Mat& img_data, uint32_t frame_num);
};
