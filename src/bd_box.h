#pragma once

// General modules
#include <cstdint>

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

        // Member functions
        BdBox();
        ~BdBox();
        bool operator==(const BdBox& a) const;
        bool operator!=(const BdBox& a) const;
        void set_parent(Person person);
        Person get_parent() const;
};
