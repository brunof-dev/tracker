#include "bd_box.h"

BdBox::BdBox() {
}

BdBox::~BdBox() {
}

bool BdBox::operator==(const BdBox& a) const {
    bool rc;
    if ((conf == a.conf) && (xmin == a.xmin) && (ymin == a.ymin) && (xmax == a.xmax) && (ymax == a.ymax) &&\
        (frame_num == a.frame_num)) rc = true;
    else rc = false;
    return(rc);
}

bool BdBox::operator!=(const BdBox& a) const {
    bool rc;
    if ((conf != a.conf) || (xmin != a.xmin) || (ymin != a.ymin) || (xmax != a.xmax) || (ymax != a.ymax) ||\
        (frame_num != a.frame_num)) rc = true;
    else rc = false;
    return(rc);
}

void BdBox::setParent(const Person& person) {
    m_person = person;
}

Person BdBox::getParent() const {
    return(m_person);
}
