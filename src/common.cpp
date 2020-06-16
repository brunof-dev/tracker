#include "common.h"

void common::fillBlob(InferenceEngine::Blob::Ptr input_blob, const cv::Mat& img_data) {
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

void common::getBdBox(InferenceEngine::Blob::Ptr output_blob, uint16_t orig_width, uint16_t orig_height, uint32_t frame_num,
                      std::vector<BdBox>& bd_box_vec) {
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
                bd_box.frame_num = frame_num;
                bd_box_vec.push_back(bd_box);
            }
        }
    }
}

float common::IOU(const BdBox& a, const BdBox& b) {
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

void common::getHOG(std::vector<BdBox>& bd_box_vec, cv::Mat img_data, const cv::HOGDescriptor& hog_handler) {
    for (std::vector<BdBox>::iterator it = bd_box_vec.begin(); it != bd_box_vec.end(); it++) {
        // Crop bounding box
        cv::Mat crop_data(img_data, cv::Rect(cv::Point2i(it->xmin, it->ymin), cv::Point2i(it->xmax, it->ymax)));

        // Resize
        cv::resize(crop_data, crop_data, hog_handler.winSize);

        // Compute HOG descriptor
        hog_handler.compute(crop_data, it->hogd);
    }
}

void common::getORB(std::vector<BdBox>& bd_box_vec, cv::Mat img_data, cv::Ptr<cv::ORB> orb_handler) {
    for (std::vector<BdBox>::iterator it = bd_box_vec.begin(); it != bd_box_vec.end(); it++) {
        // Crop bounding box
        cv::Mat crop_data(img_data, cv::Rect(cv::Point2i(it->xmin, it->ymin), cv::Point2i(it->xmax, it->ymax)));
        cv::Mat orig_data = crop_data.clone();

        // Resize
        cv::resize(crop_data, crop_data, cv::Size(120, 240));

        // Compute SIFT descriptor
        std::vector<cv::KeyPoint> keypoints;
        orb_handler->detectAndCompute(crop_data, cv::noArray(), keypoints, it->orbd);

        // Exclude very small bounding boxes (yield no keypoints and hence no ORB descriptor)
        if (it->orbd.empty()) {
            it = bd_box_vec.erase(it);
            it--;
        }
    }
}

uint32_t common::distance(const BdBox& a, const BdBox& b) {
    // Coordinates of the center of the bounding boxes
    const uint16_t a_xc = static_cast<uint16_t>((a.xmin + a.xmax) * 0.5);
    const uint16_t a_yc = static_cast<uint16_t>((a.ymin + a.ymax) * 0.5);
    const uint16_t b_xc = static_cast<uint16_t>((b.xmin + b.xmax) * 0.5);
    const uint16_t b_yc = static_cast<uint16_t>((b.ymin + b.ymax) * 0.5);

    // Distance
    uint32_t dist = (a_xc - b_xc) * (a_xc - b_xc) + (a_yc - b_yc) * (a_yc - b_yc);
    return(dist);
}

float common::distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); i++) dist += (a.at(i) - b.at(i)) * (a.at(i) - b.at(i));
    return(dist);
}

float common::distance(const cv::Mat& a, const cv::Mat& b) {
    const uint16_t row_max = std::min(a.rows, b.rows);
    double dist = 0.0;
    for (uint16_t i = 0; i < row_max; i++) {
        dist += cv::norm(a.row(i), b.row(i), cv::NormTypes::NORM_HAMMING);
    }
    return(static_cast<float>(dist));
}

void common::findNeighbour(const BdBox& bd_box, const std::vector<BdBox>& bd_box_vec, BdBox& neighbour) {
    uint32_t best_dist;
    for (std::vector<BdBox>::const_iterator it = bd_box_vec.cbegin(); it != bd_box_vec.cend(); it++) {
        // Distance between boxes
        uint32_t dist = distance(bd_box, *it);
        bool found = false;
        if (it == bd_box_vec.begin()) {
            best_dist = dist;
            found = true;
        }
        else if (dist < best_dist) {
            best_dist = dist;
            found = true;
        }
        // Max person speed threshold
        if ((best_dist < threshold::DIST) && (found == true)) neighbour = *it;
    }
}

void common::findNbVec(const BdBox& bd_box, const std::vector<BdBox>& bd_box_vec, std::vector<BdBox>& nb_vec) {
    uint32_t best_dist;
    for (std::vector<BdBox>::const_iterator it = bd_box_vec.cbegin(); it != bd_box_vec.cend(); it++) {
        // Distance between boxes
        uint32_t dist = distance(bd_box, *it);
        bool found = false;
        if (it == bd_box_vec.begin()) {
            best_dist = dist;
            found = true;
        }
        else if (dist < best_dist) {
            best_dist = dist;
            found = true;
        }
        // Max person speed threshold
        if ((best_dist < threshold::DIST) && (found == true)) nb_vec.push_back(*it);
    }
}

void common::matchNbVec(const BdBox& bd_box, const std::vector<BdBox>& nb_vec, BdBox& neighbour) {
    float best_dist;
    for (std::vector<BdBox>::const_iterator it = nb_vec.cbegin(); it != nb_vec.cend(); it++) {
        // Distance between descriptors
        float dist = distance(bd_box.hogd, it->hogd);
        // float dist = distance(bd_box.orbd, it->orbd);
        bool found = false;
        if (it == nb_vec.begin()) {
            best_dist = dist;
            found = true;
        }
        else if (dist < best_dist) {
            best_dist = dist;
            found = true;
        }
        if (found == true) neighbour = *it;
    }
}

void common::assignID(std::vector<BdBox>& bd_box_vec, const std::vector<std::vector<BdBox>>& bd_box_stack,
                      uint32_t frame_num, std::vector<Person>& person_vec) {

    for (std::vector<BdBox>::iterator i = bd_box_vec.begin(); i != bd_box_vec.end(); i++) {
        // Dummy bounding box in case no neighbour is found
        BdBox dummy;
        dummy.conf = -1.0; dummy.xmin = 0; dummy.xmax = 0; dummy.ymin = 0; dummy.ymax = 0; dummy.frame_num = 0;
        BdBox neighbour = dummy;
        std::vector<BdBox> nb_vec;

        // Neighbour search
        if (frame_num != 0) {
            for (std::vector<std::vector<BdBox>>::const_iterator j = bd_box_stack.cbegin(); j != bd_box_stack.cend(); j++) {
                findNbVec(*i, *j, nb_vec);
                if (nb_vec.size() >= threshold::NEIGHBOURS) break;
            }
            matchNbVec(*i, nb_vec, neighbour);
        }

        // Update IDs
        if (neighbour == dummy) {
            // Completely new person
            Person person(*i);
            i->setParent(person);
            person_vec.push_back(person);
        }
        else {
            // Person from previous frames
            Person person = neighbour.getParent();
            i->setParent(person);
            uint32_t id = person.getId();
            person_vec.at(id - 1).addBdBox(*i);
            person_vec.at(id - 1).enroll();
        }
    }
}

void common::FIFO(const std::vector<BdBox>& bd_box_vec, std::vector<std::vector<BdBox>>& bd_box_stack) {
    bd_box_stack.insert(bd_box_stack.begin(), bd_box_vec);
    if (bd_box_stack.size() > threshold::STACK) {
        bd_box_stack.pop_back();
    }
}

void common::show(const std::vector<Person>& person_vec, cv::Mat& img_data, uint32_t frame_num) {
    // Counting
    std::stringstream s;
    s << "People: " << Person::count;
    cv::putText(img_data, s.str(), cv::Point2i(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.75, display::COLOR);

    // Bounding boxes
    for (std::vector<Person>::const_iterator i = person_vec.cbegin(); i != person_vec.cend(); i++) {
        if (i->isEnrolled()) {
            const std::vector<BdBox>* bd_box_vec = i->getBdBoxVec();
            for (std::vector<BdBox>::const_iterator j = bd_box_vec->cbegin(); j != bd_box_vec->cend(); j++) {
                if (j->frame_num == frame_num) {
                    srand(i->getId());
                    uint8_t red = rand() % 255;
                    uint8_t green = rand() % 255;
                    uint8_t blue = rand() % 255;
                    cv::Scalar color(red, green, blue);
                    cv::putText(img_data, std::to_string(i->getId()), cv::Point2i(j->xmin, j->ymin - display::TXT_OFFSET),
                                cv::FONT_HERSHEY_SIMPLEX, 0.75, color);
                    cv::rectangle(img_data, cv::Point2i(j->xmin, j->ymin), cv::Point2i(j->xmax, j->ymax), color, 2);
                    break;
                }
            }
        }
    }

    // Play
    cv::imshow("Tracker", img_data);
    cv::waitKey(1);
}
