#pragma once

// General modules
#include <cstdint>
#include <opencv2/opencv.hpp>

// Local modules
#include "person.h"

class BdBox {

    private:
        Person m_person;

    public:
        // Attributes
        float conf;
        uint16_t xmin;
        uint16_t ymin;
        uint16_t xmax;
        uint16_t ymax;
        uint32_t frame_num;
        std::vector<float> hogd;
        cv::Mat orbd;

        // Member functions
        BdBox();
        ~BdBox();
        bool operator==(const BdBox& a) const;
        bool operator!=(const BdBox& a) const;
        void setParent(const Person& person);
        Person getParent() const;
};
