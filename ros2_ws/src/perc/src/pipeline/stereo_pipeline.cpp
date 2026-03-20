#include "perc/pipeline/stereo_pipeline.hpp"

#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/Status.h>
#include <cstdio>

StereoPipeline::StereoPipeline() = default;

StereoPipeline::~StereoPipeline()
{
    destroyViews();

    if (nv12_cuda_) {
        vpiImageDestroy(nv12_cuda_);
        nv12_cuda_ = nullptr;
    }

    if (stream_) {
        vpiStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void StereoPipeline::destroyViews()
{
    for (auto &q : quad_) {
        if (q) {
            vpiImageDestroy(q);
            q = nullptr;
        }
    }
}

static void printVPIError(const char *where, VPIStatus st)
{
    char msg[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
    vpiPeekAtLastStatusMessage(msg, sizeof(msg));
    std::fprintf(stderr, "%s failed: %s (%d): %s\n",
                 where, vpiStatusGetName(st), st, msg);
}

bool StereoPipeline::init(int width, int height)
{
    width_ = width;
    height_ = height;

    half_w_ = (width_ / 2) & ~1;
    half_h_ = (height_ / 2) & ~1;

    if (vpiStreamCreate(VPI_BACKEND_CUDA, &stream_) != VPI_SUCCESS) {
        std::fprintf(stderr, "vpiStreamCreate failed\n");
        return false;
    }

    // Create a CUDA-view-compatible NV12 image.
    // Views on Tegra require parent image to wrap CPU/CUDA buffer
    // or have only CPU/CUDA backends enabled.
    VPIStatus st = vpiImageCreate(
        width_,
        height_,
        VPI_IMAGE_FORMAT_NV12_ER_BL,
        VPI_BACKEND_CUDA,
        &nv12_cuda_);

    if (st != VPI_SUCCESS) {
        std::fprintf(stderr, "vpiImageCreate(nv12_cuda_) failed, status=%d\n", st);
        return false;
    }

    roi_[0] = {0,       0,       half_w_, half_h_}; // TL
    roi_[1] = {half_w_, 0,       half_w_, half_h_}; // TR
    roi_[2] = {0,       half_h_, half_w_, half_h_}; // BL
    roi_[3] = {half_w_, half_h_, half_w_, half_h_}; // BR

    if (!createQuadrantViews()) {
        return false;
    }

    return true;
}

bool StereoPipeline::createQuadrantViews()
{
    for (int i = 0; i < 4; ++i) {
        VPIStatus st = vpiImageCreateView(nv12_cuda_, &roi_[i], 0, &quad_[i]);
        if (st != VPI_SUCCESS) {
            std::fprintf(stderr, "quad %d: ", i);
            printVPIError("vpiImageCreateView", st);
            return false;
        }
    }
    return true;
}

bool StereoPipeline::updateCudaParent(const GpuFrame& frame)
{
    // Copy/convert wrapped NVBUFFER NV12 into CUDA-only NV12 parent.
    VPIStatus st = vpiSubmitConvertImageFormat(
        stream_,
        VPI_BACKEND_CUDA,
        frame.nv12,
        nv12_cuda_,
        nullptr);

    if (st != VPI_SUCCESS) {
        printVPIError("vpiSubmitConvertImageFormat", st);
        return false;
    }

    st = vpiStreamSync(stream_);
    if (st != VPI_SUCCESS) {
        printVPIError("vpiStreamSync", st);
        return false;
    }

    return true;
}

bool StereoPipeline::process(const GpuFrame& frame)
{
    if (!frame.nv12) {
        return false;
    }

    if (!updateCudaParent(frame)) {
        return false;
    }

    // quad_[0..3] already point into nv12_cuda_
    return true;
}