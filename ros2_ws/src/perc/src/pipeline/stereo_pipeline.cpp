#include "perc/pipeline/stereo_pipeline.hpp"

#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Remap.h>
#include <vpi/Status.h>

#include <cstdio>
#include <cstring>

StereoPipeline::StereoPipeline() = default;

static void printVPIError(const char *where, VPIStatus st)
{
    char msg[VPI_MAX_STATUS_MESSAGE_LENGTH] = {};
    vpiPeekAtLastStatusMessage(msg, sizeof(msg));
    std::fprintf(stderr, "%s failed: %s (%d): %s\n",
                 where, vpiStatusGetName(st), st, msg);
}

StereoPipeline::~StereoPipeline()
{
    destroyViews();
    destroyPayloads();
    destroyWarpMaps();
    destroyImages();

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

    for (auto &g : gray_view_) {
        if (g) {
            vpiImageDestroy(g);
            g = nullptr;
        }
    }
}

void StereoPipeline::destroyImages()
{
    for (auto &r : rect_) {
        if (r) {
            vpiImageDestroy(r);
            r = nullptr;
        }
    }

    if (y_full_) {
        vpiImageDestroy(y_full_);
        y_full_ = nullptr;
    }

    if (nv12_parent_) {
        vpiImageDestroy(nv12_parent_);
        nv12_parent_ = nullptr;
    }
}

void StereoPipeline::destroyWarpMaps()
{
    for (auto &w : warp_) {
        if (w.keypoints) {
            vpiWarpMapFreeData(&w);
        }
        std::memset(&w, 0, sizeof(w));
    }
}

void StereoPipeline::destroyPayloads()
{
    for (auto &p : remap_payload_) {
        if (p) {
            vpiPayloadDestroy(p);
            p = nullptr;
        }
    }
}

bool StereoPipeline::createParentBuffers()
{
    VPIStatus st = vpiImageCreate(
        width_,
        height_,
        VPI_IMAGE_FORMAT_NV12_ER,
        VPI_BACKEND_CUDA,
        &nv12_parent_);

    if (st != VPI_SUCCESS) {
        printVPIError("vpiImageCreate(nv12_parent_)", st);
        return false;
    }

    st = vpiImageCreate(
        width_,
        height_,
        VPI_IMAGE_FORMAT_Y8_ER,
        VPI_BACKEND_CUDA,
        &y_full_);

    if (st != VPI_SUCCESS) {
        printVPIError("vpiImageCreate(y_full_)", st);
        return false;
    }

    return true;
}

bool StereoPipeline::createRectBuffers()
{
    for (int i = 0; i < 4; ++i) {
        VPIStatus st = vpiImageCreate(
            half_w_,
            half_h_,
            VPI_IMAGE_FORMAT_Y8_ER,
            VPI_BACKEND_CUDA,
            &rect_[i]);

        if (st != VPI_SUCCESS) {
            printVPIError("vpiImageCreate(rect_)", st);
            return false;
        }
    }
    return true;
}

bool StereoPipeline::createGrayViews()
{
    for (int i = 0; i < 4; ++i) {
        VPIStatus st = vpiImageCreateView(y_full_, &roi_[i], 0, &gray_view_[i]);
        if (st != VPI_SUCCESS) {
            std::fprintf(stderr, "gray view %d: ", i);
            printVPIError("vpiImageCreateView(y_full_)", st);
            return false;
        }
    }
    return true;
}

static bool fillWarpFromMaps(const cv::Mat& map_x,
                             const cv::Mat& map_y,
                             VPIWarpMap& warp)
{
    if (map_x.empty() || map_y.empty()) {
        return false;
    }
    if (map_x.rows != map_y.rows || map_x.cols != map_y.cols) {
        return false;
    }

    std::fprintf(stderr, "Map dimensions: %dx%d\n", map_x.cols, map_x.rows);
    std::fflush(stderr);

    std::fprintf(stderr, "map x type: %d, map y type: %d\n ", map_x.type(), map_y.type());
    std::fflush(stderr);

    std::memset(&warp, 0, sizeof(warp));
    warp.grid.numHorizRegions = 1;
    warp.grid.numVertRegions  = 1;
    warp.grid.regionWidth[0]  = static_cast<int16_t>(map_x.cols);
    warp.grid.regionHeight[0] = static_cast<int16_t>(map_x.rows);
    warp.grid.horizInterval[0] = 1;
    warp.grid.vertInterval[0]  = 1;

    VPIStatus st = vpiWarpMapAllocData(&warp);
    if (st != VPI_SUCCESS) {
        printVPIError("vpiWarpMapAllocData", st);
        return false;
    }

    std::fprintf(stderr, "Allocated warp map data horiz=%d vert=%d pitch=%d\n",
                warp.numHorizPoints, warp.numVertPoints, warp.pitchBytes);
    std::fflush(stderr);

    // Initialize padded control points safely
    st = vpiWarpMapGenerateIdentity(&warp);
    if (st != VPI_SUCCESS) {
        printVPIError("vpiWarpMapGenerateIdentity", st);
        return false;
    }

    std::fprintf(stderr, "Filling warp map data\n");
    std::fflush(stderr);

    // Only fill the overlapping valid region from calibration maps
    for (int y = 0; y < map_x.rows; ++y) {
        auto *row = reinterpret_cast<VPIKeypointF32*>(
            reinterpret_cast<uint8_t*>(warp.keypoints) + y * warp.pitchBytes);

        for (int x = 0; x < map_x.cols; ++x) {
            row[x].x = map_x.at<float>(y, x);
            row[x].y = map_y.at<float>(y, x);
        }
    }

    return true;
}

bool StereoPipeline::createWarpMaps(const DualStereoCalibration& calib)
{
    return fillWarpFromMaps(calib.pair0.left_map_x,  calib.pair0.left_map_y,  warp_[0]) &&
           fillWarpFromMaps(calib.pair0.right_map_x, calib.pair0.right_map_y, warp_[1]) &&
           fillWarpFromMaps(calib.pair1.left_map_x,  calib.pair1.left_map_y,  warp_[2]) &&
           fillWarpFromMaps(calib.pair1.right_map_x, calib.pair1.right_map_y, warp_[3]);
}

bool StereoPipeline::createRemapPayloads()
{
    for (int i = 0; i < 4; ++i) {
        VPIStatus st = vpiCreateRemap(VPI_BACKEND_CUDA, &warp_[i], &remap_payload_[i]);
        if (st != VPI_SUCCESS) {
            printVPIError("vpiCreateRemap", st);
            return false;
        }
    }
    return true;
}

bool StereoPipeline::init(int width, int height, const DualStereoCalibration& calib)
{
    width_ = width;
    height_ = height;

    half_w_ = (width_ / 2) & ~1;
    half_h_ = (height_ / 2) & ~1;

    std::fprintf(stderr, "Frame Size %dx%d\n", half_w_, half_h_);
    std::fflush(stderr);

    VPIStatus st = vpiStreamCreate(VPI_BACKEND_CUDA, &stream_);
    if (st != VPI_SUCCESS) {
        printVPIError("vpiStreamCreate", st);
        return false;
    }

    std::fprintf(stderr, "[StereoPipeline::init] createParentBuffers\n");
    std::fflush(stderr);
    if (!createParentBuffers()) return false;

    roi_[0] = {0,       0,       half_w_, half_h_};
    roi_[1] = {half_w_, 0,       half_w_, half_h_};
    roi_[2] = {0,       half_h_, half_w_, half_h_};
    roi_[3] = {half_w_, half_h_, half_w_, half_h_};

    std::fprintf(stderr, "[StereoPipeline::init] createQuadrantViews\n");
    std::fflush(stderr);
    if (!createQuadrantViews()) return false;

    std::fprintf(stderr, "[StereoPipeline::init] createGrayViews\n");
    std::fflush(stderr);
    if (!createGrayViews()) return false;

    std::fprintf(stderr, "[StereoPipeline::init] createRectBuffers\n");
    std::fflush(stderr);
    if (!createRectBuffers()) return false;

    std::fprintf(stderr, "[StereoPipeline::init] DONE\n");
    std::fflush(stderr);

    return true;
}

bool StereoPipeline::createQuadrantViews()
{
    for (int i = 0; i < 4; ++i) {
        VPIStatus st = vpiImageCreateView(nv12_parent_, &roi_[i], 0, &quad_[i]);
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
    VPIStatus st = vpiSubmitConvertImageFormat(
        stream_,
        VPI_BACKEND_CUDA,
        frame.nv12,
        nv12_parent_,
        nullptr);

    if (st != VPI_SUCCESS) {
        printVPIError("vpiSubmitConvertImageFormat(frame->nv12_cuda_)", st);
        return false;
    }

    st = vpiStreamSync(stream_);
    if (st != VPI_SUCCESS) {
        printVPIError("vpiStreamSync(updateCudaParent)", st);
        return false;
    }

    return true;
}

bool StereoPipeline::convertFullFrameToGray()
{
    std::fprintf(stderr, "[convertFullFrameToGray] nv12_parent_ -> y_full_\n");
    std::fflush(stderr);

    VPIStatus st = vpiSubmitConvertImageFormat(
        stream_,
        VPI_BACKEND_CUDA,
        nv12_parent_,
        y_full_,
        nullptr);

    if (st != VPI_SUCCESS) {
        printVPIError("vpiSubmitConvertImageFormat(nv12_parent_->y_full_)", st);
        return false;
    }

    st = vpiStreamSync(stream_);
    if (st != VPI_SUCCESS) {
        printVPIError("vpiStreamSync(convertFullFrameToGray)", st);
        return false;
    }

    return true;
}

bool StereoPipeline::rectifyQuadrants()
{
    for (int i = 0; i < 4; ++i) {
        VPIStatus st = vpiSubmitRemap(
            stream_,
            VPI_BACKEND_CUDA,
            remap_payload_[i],
            gray_view_[i],
            rect_[i],
            VPI_INTERP_LINEAR,
            VPI_BORDER_ZERO,
            0);

        if (st != VPI_SUCCESS) {
            printVPIError("vpiSubmitRemap", st);
            return false;
        }
    }

    VPIStatus st = vpiStreamSync(stream_);
    if (st != VPI_SUCCESS) {
        printVPIError("vpiStreamSync(rectifyQuadrants)", st);
        return false;
    }

    return true;
}

bool StereoPipeline::process(const GpuFrame& frame)
{
    if (!frame.nv12) {
        return false;
    }

    std::fprintf(stderr, "[StereoPipeline::process] updateCudaParent\n");
    std::fflush(stderr);
    if (!updateCudaParent(frame)) return false;

    std::fprintf(stderr, "[StereoPipeline::process] convertFullFrameToGray\n");
    std::fflush(stderr);
    if (!convertFullFrameToGray()) return false;

    std::fprintf(stderr, "[StereoPipeline::process] rectifyQuadrants\n");
    std::fflush(stderr);
    if (!rectifyQuadrants()) return false;

    std::fprintf(stderr, "[StereoPipeline::process] done\n");
    std::fflush(stderr);
    return true;

    return true;
}