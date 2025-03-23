#include "profiler.h"

// Define the static variables
namespace Profiler {
    float KernelTimeTracker::last_kernel_time = 0.0f;
    float KernelTimeTracker::last_total_time = 0.0f;
}
