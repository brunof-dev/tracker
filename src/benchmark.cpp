#include "benchmark.h"

Benchmark::Benchmark() {
}

Benchmark::~Benchmark() {
}

void Benchmark::start() {
    m_start = std::chrono::system_clock::now();
}

void Benchmark::end(const char* const msg) {
    m_end = std::chrono::system_clock::now();
    m_elapsed = m_end - m_start;
    // std::cout << msg << " time: " << m_elapsed.count() << "ms" << std::endl;
}
