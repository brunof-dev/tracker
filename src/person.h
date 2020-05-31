#pragma once

// General modules
#include <vector>
#include <cstdint>

class BdBox;

class Person {

    private:
        std::vector<BdBox> m_bd_box_vec;
        uint32_t m_id;

    public:
        // Static member
        static uint32_t count;

        // Constructor
        Person();
        Person(const BdBox& bd_box);
        ~Person();

        // Getters
        const std::vector<BdBox>* get_bd_box_vec() const;
        uint32_t get_id() const;

        // Setters
        void add_bd_box(const BdBox& bd_box);
        void set_id(uint32_t id);
};

// Local modules
#include "bd_box.h"
