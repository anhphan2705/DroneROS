// nvmm_cuda_bridge.c
#include <gst/gst.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda.h>
#include <cudaEGL.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

// ---- Minimal POD to hand back to Python ----
typedef struct {
    CUgraphicsResource resource;    // must be released
    NvBufSurface *surf;             // not owned; mapped for this call
    // From cudaEglFrame
    CUdeviceptr y_ptr;
    size_t      y_pitch;
    CUdeviceptr uv_ptr;
    size_t      uv_pitch;
    int width;
    int height;
} CudaNv12Frame;

// Return 0 on success. On success, out->resource must be released via nvmm_cuda_unmap().
__attribute__((visibility("default")))
int nvmm_cuda_map_from_gstbuffer(GstBuffer *buf, CudaNv12Frame *out) {
    if (!buf || !out) return -1;

    // Argus/GStreamer puts NvBufSurface* directly in mapped data for NVMM buffers
    GstMapInfo map_info;
    if (!gst_buffer_map(buf, &map_info, GST_MAP_READ)) return -2;

    NvBufSurface *surf = (NvBufSurface*)map_info.data;
    gst_buffer_unmap(buf, &map_info);
    if (!surf) return -3;

    // Make EGLImage
    if (NvBufSurfaceMapEglImage(surf, 0) != 0) return -4;
    EGLImageKHR eglImage = surf->surfaceList[0].mappedAddr.eglImage;
    if (!eglImage) { NvBufSurfaceUnMapEglImage(surf, 0); return -5; }

    // Register EGL image with CUDA
    CUresult cuErr;
    CUgraphicsResource resource = nullptr;
    cuErr = cuInit(0);
    if (cuErr != CUDA_SUCCESS) { NvBufSurfaceUnMapEglImage(surf, 0); return -6; }

    cuErr = cuGraphicsEGLRegisterImage(&resource, eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (cuErr != CUDA_SUCCESS) { NvBufSurfaceUnMapEglImage(surf, 0); return -7; }

    // Get mapped EGL frame
    cudaEglFrame eglFrame;
    CUeglColorFormat cf;
    cuErr = cuGraphicsResourceGetMappedEglFrame(&eglFrame, resource, 0, 0);
    if (cuErr != CUDA_SUCCESS) {
        cuGraphicsUnregisterResource(resource);
        NvBufSurfaceUnMapEglImage(surf, 0);
        return -8;
    }

    // Expect NV12: 2 pitched pointers (Y and UV)
    if (eglFrame.frameType != cudaEglFrameTypePitch || eglFrame.planeCount < 2) {
        cuGraphicsUnregisterResource(resource);
        NvBufSurfaceUnMapEglImage(surf, 0);
        return -9;
    }

    // Plane 0: Y; Plane 1: UV interleaved
    out->y_ptr   = (CUdeviceptr)eglFrame.frame.pPitch[0];
    out->y_pitch = eglFrame.pitch;
    out->uv_ptr  = (CUdeviceptr)eglFrame.frame.pPitch[1];
    out->uv_pitch= eglFrame.pitch;
    out->width   = eglFrame.width;
    out->height  = eglFrame.height;
    out->resource= resource;
    out->surf    = surf;

    return 0;
}

// Always call this to clean up
__attribute__((visibility("default")))
void nvmm_cuda_unmap(CudaNv12Frame *frame) {
    if (!frame) return;
    if (frame->resource) {
        cuGraphicsUnregisterResource(frame->resource);
        frame->resource = nullptr;
    }
    if (frame->surf) {
        NvBufSurfaceUnMapEglImage(frame->surf, 0);
        frame->surf = NULL;
    }
}

// Simple GPU kernel to extract a ROI (top-left corner x,y; width w; height h) from NV12 → planar Y + UV copy.
// You can replace this with VPI later, but this keeps everything on GPU.
__global__ void copy_nv12_roi(
    const unsigned char* __restrict__ y, size_t y_pitch,
    const unsigned char* __restrict__ uv, size_t uv_pitch,
    int src_w, int src_h,
    int roi_x, int roi_y, int roi_w, int roi_h,
    unsigned char* __restrict__ out_y, size_t out_y_pitch,
    unsigned char* __restrict__ out_uv, size_t out_uv_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y_i = blockIdx.y * blockDim.y + threadIdx.y; // row

    // Copy Y
    if (x < roi_w && y_i < roi_h) {
        const unsigned char *srcY = y + (roi_y + y_i) * y_pitch + (roi_x + x);
        unsigned char *dstY = out_y + y_i * out_y_pitch + x;
        *dstY = *srcY;
    }

    // Copy UV (every 2x2 pixel)
    if (x < roi_w/2 && y_i < roi_h/2) {
        const unsigned char *srcUV = uv + ((roi_y/2) + y_i) * uv_pitch + (roi_x) + (x*2);
        unsigned char *dstUV = out_uv + y_i * out_uv_pitch + (x*2);
        dstUV[0] = srcUV[0];
        dstUV[1] = srcUV[1];
    }
}

// Allocate device buffers for an NV12 image (Y + UV) and copy an ROI from src to dst.
// Returns 0 on success; outputs device pointers + pitches for the ROI image.
__attribute__((visibility("default")))
int nv12_extract_roi_cuda(
    CUdeviceptr src_y, size_t src_y_pitch,
    CUdeviceptr src_uv, size_t src_uv_pitch,
    int src_w, int src_h,
    int roi_x, int roi_y, int roi_w, int roi_h,
    CUdeviceptr *out_y, size_t *out_y_pitch,
    CUdeviceptr *out_uv, size_t *out_uv_pitch)
{
    if (!out_y || !out_y_pitch || !out_uv || !out_uv_pitch) return -1;
    CUresult err;

    // Allocate pitched memory for destination (NV12)
    err = cuMemAllocPitch(out_y, out_y_pitch, roi_w, roi_h, 16);
    if (err != CUDA_SUCCESS) return -2;
    err = cuMemAllocPitch(out_uv, out_uv_pitch, roi_w, roi_h/2, 16);
    if (err != CUDA_SUCCESS) { cuMemFree(*out_y); return -3; }

    dim3 block(16, 16);
    dim3 grid((roi_w + block.x - 1)/block.x, (roi_h + block.y - 1)/block.y);

    copy_nv12_roi<<<grid, block>>>(
        (const unsigned char*)src_y, src_y_pitch,
        (const unsigned char*)src_uv, src_uv_pitch,
        src_w, src_h,
        roi_x, roi_y, roi_w, roi_h,
        (unsigned char*)(*out_y), *out_y_pitch,
        (unsigned char*)(*out_uv), *out_uv_pitch
    );
    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        cuMemFree(*out_y);
        cuMemFree(*out_uv);
        return -4;
    }

    return 0;
}

// Free device buffers allocated by nv12_extract_roi_cuda
__attribute__((visibility("default")))
void nv12_free_cuda(CUdeviceptr ptrY, CUdeviceptr ptrUV) {
    if (ptrY) cuMemFree(ptrY);
    if (ptrUV) cuMemFree(ptrUV);
}
