#include "person.h"

uint32_t Person::count = 0;

Person::Person() {
}

Person::Person(const BdBox& bd_box) {
    // Update instance counter
    count++;

    // Assign private members
    m_id = count;
    m_bd_box_vec.push_back(bd_box);
}

Person::~Person() {
}

const std::vector<BdBox>* Person::get_bd_box_vec() const {
    return(&m_bd_box_vec);
}

uint32_t Person::get_id() const {
    return(m_id);
}

void Person::add_bd_box(const BdBox& bd_box) {
    m_bd_box_vec.push_back(bd_box);
}

void Person::set_id(uint32_t id) {
    m_id = id;
}
