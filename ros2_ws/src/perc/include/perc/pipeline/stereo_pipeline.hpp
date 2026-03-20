#pragma once

#include "perc/common/gpu_frame.hpp"

#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>

class StereoPipeline
{
public:
    StereoPipeline();
    ~StereoPipeline();

    bool init(int width, int height);
    bool process(const GpuFrame& frame);

private:
    int width_{0};
    int height_{0};
    int half_w_{0};
    int half_h_{0};

    VPIStream stream_{nullptr};

    // CUDA-only parent image for quadrant views
    VPIImage nv12_cuda_{nullptr};

    VPIRectangleI roi_[4]{};
    VPIImage quad_[4] = {nullptr, nullptr, nullptr, nullptr};

    bool createQuadrantViews();
    bool updateCudaParent(const GpuFrame& frame);
    void destroyViews();
};