#pragma once

#include <cstdint>
#include <gst/gst.h>
#include <vpi/VPI.h>

struct GpuFrame
{
    // Wrapped camera frame (NVMM -> VPI)
    VPIImage nv12 = nullptr;

    // Keep underlying Gst buffer/sample alive while nv12 wrapper is in use
    GstSample* sample = nullptr;

    int width = 0;
    int height = 0;

    uint64_t timestamp_ns = 0;
};

inline void releaseGpuFrame(GpuFrame& frame)
{
    if (frame.nv12) {
        vpiImageDestroy(frame.nv12);
        frame.nv12 = nullptr;
    }

    if (frame.sample) {
        gst_sample_unref(frame.sample);
        frame.sample = nullptr;
    }

    frame.width = 0;
    frame.height = 0;
    frame.timestamp_ns = 0;
}