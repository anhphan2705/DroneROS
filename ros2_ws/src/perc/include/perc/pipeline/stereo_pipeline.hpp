#pragma once

#include "perc/common/gpu_frame.hpp"
#include "perc/common/calibration_data.hpp"

#include <opencv2/core.hpp>
#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/WarpMap.h>

class StereoPipeline
{
public:
    StereoPipeline();
    ~StereoPipeline();

    bool init(int width, int height, const DualStereoCalibration& calib);
    bool process(const GpuFrame& frame);

    VPIImage rectified(int idx) const { return rect_[idx]; }

private:
    int width_{0};
    int height_{0};
    int half_w_{0};
    int half_h_{0};

    VPIStream stream_{nullptr};

    VPIImage nv12_parent_{nullptr};
    VPIImage y_full_{nullptr};
    VPIImage rect_[4] = {nullptr, nullptr, nullptr, nullptr};

    VPIRectangleI roi_[4]{};

    VPIWarpMap warp_[4]{};
    VPIPayload remap_payload_[4] = {nullptr, nullptr, nullptr, nullptr};

    bool createParentBuffers();
    bool createRectBuffers();
    bool createWarpMaps(const DualStereoCalibration& calib);
    bool createRemapPayloads();

    bool updateCudaParent(const GpuFrame& frame);
    bool convertFullFrameToGray();
    bool rectifyQuadrants();

    void destroyViews();
    void destroyImages();
    void destroyWarpMaps();
    void destroyPayloads();
};