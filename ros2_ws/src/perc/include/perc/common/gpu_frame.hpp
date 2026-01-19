#pragma once

#include <vpi/VPI.h>
#include <cstdint>

struct GpuFrame
{
    // Original camera frame (NVMM → EGL → VPI)
    VPIImage nv12 = nullptr;

    // Converted color (for rectification / detection)
    VPIImage bgr = nullptr;

    // Rectified quadrants (GPU)
    VPIImage rect[4] = {nullptr, nullptr, nullptr, nullptr};

    int width = 0;
    int height = 0;

    uint64_t timestamp_ns = 0;
};
