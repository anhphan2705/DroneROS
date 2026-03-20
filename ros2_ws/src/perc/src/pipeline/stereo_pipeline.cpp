#include "perc/pipeline/stereo_pipeline.hpp"

#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
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

// static void inspectVPIImage(VPIImage img, const char *name)
// {
//     VPIStatus st;

//     int32_t w = 0, h = 0;
//     VPIImageFormat fmt = 0;
//     uint64_t flags = 0;

//     st = vpiImageGetSize(img, &w, &h);
//     if (st != VPI_SUCCESS) {
//         printVPIError("vpiImageGetSize", st);
//         return;
//     }

//     st = vpiImageGetFormat(img, &fmt);
//     if (st != VPI_SUCCESS) {
//         printVPIError("vpiImageGetFormat", st);
//         return;
//     }

//     st = vpiImageGetFlags(img, &flags);
//     if (st != VPI_SUCCESS) {
//         printVPIError("vpiImageGetFlags", st);
//         return;
//     }

//     std::fprintf(stderr,
//                  "[%s] size=%dx%d format=0x%llx flags=0x%llx\n",
//                  name, w, h,
//                  static_cast<unsigned long long>(fmt),
//                  static_cast<unsigned long long>(flags));
// }

// static void inspectBackingType(VPIImage img, const char *name)
// {
//     VPIImageData data{};
//     VPIStatus st = vpiImageLockData(img, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data);
//     if (st == VPI_SUCCESS) {
//         std::fprintf(stderr, "[%s] lock as HOST_PITCH_LINEAR succeeded, bufferType=%d\n",
//                      name, data.bufferType);
//         vpiImageUnlock(img);
//         return;
//     }

//     st = vpiImageLockData(img, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &data);
//     if (st == VPI_SUCCESS) {
//         std::fprintf(stderr, "[%s] lock as CUDA_PITCH_LINEAR succeeded, bufferType=%d\n",
//                      name, data.bufferType);
//         vpiImageUnlock(img);
//         return;
//     }

//     st = vpiImageLockData(img, VPI_LOCK_READ, VPI_IMAGE_BUFFER_NVBUFFER, &data);
//     if (st == VPI_SUCCESS) {
//         std::fprintf(stderr, "[%s] lock as NVBUFFER succeeded, bufferType=%d fd=%d\n",
//                      name, data.bufferType, data.buffer.fd);
//         vpiImageUnlock(img);
//         return;
//     }

//     printVPIError("vpiImageLockData", st);
// }

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
        VPI_IMAGE_FORMAT_NV12,
        VPI_BACKEND_CUDA,
        &nv12_cuda_
    );

    if (st != VPI_SUCCESS) {
        printVPIError("vpiImageCreate(nv12_cuda_)", st);
        return false;
    }

    // inspectVPIImage(nv12_cuda_, "nv12_cuda_");
    // inspectBackingType(nv12_cuda_, "nv12_cuda_");

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

    // static bool printed_once = false;
    // if (!printed_once) {
    //     inspectVPIImage(frame.nv12, "frame.nv12");
    //     inspectBackingType(frame.nv12, "frame.nv12");

    //     inspectVPIImage(nv12_cuda_, "nv12_cuda_");
    //     inspectBackingType(nv12_cuda_, "nv12_cuda_");

    //     printed_once = true;
    // }

    if (!updateCudaParent(frame)) {
        return false;
    }

    return true;
}