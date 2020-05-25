#include "common.h"

void common::fillBlob(InferenceEngine::Blob::Ptr input_blob, cv::Mat img_data) {
    // Memory management
    InferenceEngine::MemoryBlob::Ptr mem_input_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(input_blob);
    InferenceEngine::LockedMemory<void> mem_input_blob_hold = mem_input_blob->wmap();
    uint8_t* input_blob_data = mem_input_blob_hold.as<uint8_t*>();

    // Fill blob
    uint16_t width = img_data.cols;
    uint16_t height = img_data.rows;
    uint8_t channels = img_data.channels();
    for (size_t c = 0; c < channels; c++) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                input_blob_data[c * width * height + h * width + w] = img_data.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

void common::getBdBox(std::vector<BdBox>& bd_box_vec, InferenceEngine::Blob::Ptr output_blob, uint16_t orig_width,
                      uint16_t orig_height) {
    // Memory management
    InferenceEngine::MemoryBlob::Ptr mem_output_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
    InferenceEngine::LockedMemory<const void> mem_output_blob_hold = mem_output_blob->rmap();
    const float* output_blob_data = mem_output_blob_hold.as<const float*>();

    // Get blob dimensions
    std::vector<size_t> output_blob_sz = output_blob->getTensorDesc().getDims();
    const size_t prop_num = output_blob_sz.at(2);
    const size_t prop_len  = output_blob_sz.at(3);

    // Iterate through proposals
    for (size_t i = 0; i < prop_num; i++) {
        // Parse info
        size_t index = i * prop_len;
        const int16_t img_id = static_cast<int16_t>(output_blob_data[index]);
        if (img_id == model::END_CODE) break;
        const uint16_t class_id = static_cast<uint16_t>(output_blob_data[index + 1]);
        if (class_id == model::HUMAN) {
            const float conf = output_blob_data[index + 2];
            if (conf > threshold::CONF) {
                BdBox bd_box;
                bd_box.conf = conf;
                bd_box.xmin = static_cast<uint16_t>(output_blob_data[index + 3] * orig_width);
                bd_box.ymin = static_cast<uint16_t>(output_blob_data[index + 4] * orig_height);
                bd_box.xmax = static_cast<uint16_t>(output_blob_data[index + 5] * orig_width);
                bd_box.ymax = static_cast<uint16_t>(output_blob_data[index + 6] * orig_height);
                bd_box_vec.push_back(bd_box);
            }
        }
    }
}

float common::IOU(BdBox a, BdBox b) {
    float iou;
    // Assert correctness
    assert(a.xmin < a.xmax);
    assert(a.ymin < a.ymax);
    assert(b.xmin < b.xmax);
    assert(b.ymin < b.ymax);

    // Coordinates of intersection rectangle
    const uint16_t x_left = std::max(a.xmin, b.xmin);
    const uint16_t x_right = std::min(a.xmax, b.xmax);
    const uint16_t y_top = std::max(a.ymin, b.ymin);
    const uint16_t y_bottom = std::min(a.ymax, b.ymax);

    // Case of no intersection
    if ((x_right < x_left) || (y_bottom < y_top)) {
        iou = 0.0;
    }
    else {
        // Calculate areas and ratio
        const uint32_t i_area = (x_right - x_left + 1) * (y_bottom - y_top + 1);
        const uint32_t a_area = (a.xmax - a.xmin + 1) * (a.ymax - a.ymin + 1);
        const uint32_t b_area = (b.xmax - b.xmin + 1) * (b.ymax - b.ymin + 1);
        iou = static_cast<float>(i_area) / static_cast<float>(a_area + b_area - i_area);
    }
    return(iou);
}

void common::nonMaxSup(std::vector<BdBox>& bd_box_vec) {
    for (std::vector<BdBox>::iterator i = bd_box_vec.begin(); i != bd_box_vec.end(); i++) {
        // Big box suppression
        uint16_t width = i->xmax - i->xmin;
        uint16_t height = i->ymax - i->ymin;
        if ((width > threshold::WIDTH) || (height > threshold::HEIGHT)) {
            i = bd_box_vec.erase(i);
            i--;
        }
        else {
            for (std::vector<BdBox>::iterator j = bd_box_vec.begin(); j != bd_box_vec.end(); j++) {
                // Non-max suppression
                float iou = IOU(*i, *j);
                if (iou > threshold::IOU) {
                    if (i->conf > j->conf) {
                        j = bd_box_vec.erase(j);
                        j--;
                    }
                }
            }
        }
    }
}

BdBox common::findNeighbour(BdBox bd_box, cv::Mat img_data) {
}
