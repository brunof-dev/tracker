#pragma once

// General modules
#include <vector>
#include <cstdint>

// Local modules
#include "threshold.h"

class BdBox;

class Person {

    private:
        std::vector<BdBox> m_bd_box_vec;
        uint32_t m_id;
        bool m_enroll;

    public:
        // Static member
        static uint32_t count;
        static uint32_t id;

        // Constructors
        Person();
        Person(const BdBox& bd_box);

        // Destructor
        ~Person();

        // Getters
        const std::vector<BdBox>* getBdBoxVec() const;
        uint32_t getId() const;
        bool isEnrolled() const;

        // Setters
        void addBdBox(const BdBox& bd_box);
        void setId(uint32_t id);
        void enroll();
};

// Local modules
#include "bd_box.h"
