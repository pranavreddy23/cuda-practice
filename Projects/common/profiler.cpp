#include "profiler.h"

namespace Profiler {
    // Initialize static members
    std::map<std::string, float> KernelTimeTracker::kernel_times;
    float KernelTimeTracker::last_total_time = 0.0f;
}
