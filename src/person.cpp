#include "person.h"

uint32_t Person::count = 0;
uint32_t Person::id = 0;

// Constructors
Person::Person() {
}

Person::Person(const BdBox& bd_box) {
    // Update unique ID
    id++;

    // Assign private members
    m_bd_box_vec.push_back(bd_box);
    m_id = id;
    m_enroll = false;
}

Person::~Person() {
}

// Getters
const std::vector<BdBox>* Person::getBdBoxVec() const {
    return(&m_bd_box_vec);
}

uint32_t Person::getId() const {
    return(m_id);
}

bool Person::isEnrolled() const {
    return(m_enroll);
}

// Setters
void Person::addBdBox(const BdBox& bd_box) {
    m_bd_box_vec.push_back(bd_box);
}

void Person::setId(uint32_t id) {
    m_id = id;
}

void Person::enroll() {
    if ((m_bd_box_vec.size() >= threshold::OCURRENCE) && (!m_enroll)) {
        m_enroll = true;
        count++;
        m_id = count;
    }
}
