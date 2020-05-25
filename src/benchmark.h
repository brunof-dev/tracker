#pragma once

#include <chrono>
#include <iostream>

class Benchmark {

    private:
        std::chrono::time_point<std::chrono::system_clock> m_start;
        std::chrono::time_point<std::chrono::system_clock> m_end;
        std::chrono::duration<double, std::milli> m_elapsed;
    
    public:
        Benchmark();
        ~Benchmark();
        void start();
        void end(const char* const msg = "elapsed");
};
