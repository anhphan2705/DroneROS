#include "perc/camera/camera_capture.hpp"

#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>

#include <chrono>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

CameraCapture::CameraCapture() = default;

CameraCapture::~CameraCapture()
{
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
    }

    if (egl_display_ != EGL_NO_DISPLAY) {
        eglTerminate(egl_display_);
    }
}

bool CameraCapture::init(int sensor_id, int width, int height, int fps)
{
    sensor_id_ = sensor_id;
    width_ = width;
    height_ = height;
    fps_ = fps;

    egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display_ == EGL_NO_DISPLAY) return false;
    if (!eglInitialize(egl_display_, nullptr, nullptr)) return false;

    return initGStreamer();
}

bool CameraCapture::initGStreamer()
{
    gst_init(nullptr, nullptr);

    std::string pipeline_str =
        "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id_) + " ! "
        "video/x-raw(memory:NVMM),width=" + std::to_string(width_) +
        ",height=" + std::to_string(height_) +
        ",framerate=" + std::to_string(fps_) + "/1,format=NV12 ! "
        "appsink name=nvmm_sink max-buffers=1 drop=true sync=false";

    pipeline_ = gst_parse_launch(pipeline_str.c_str(), nullptr);
    if (!pipeline_) return false;

    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline_), "nvmm_sink");
    if (!sink) return false;

    appsink_ = GST_APP_SINK(sink);
    gst_object_unref(sink);

    if (!appsink_) return false;

    gst_app_sink_set_emit_signals(appsink_, FALSE);
    gst_app_sink_set_drop(appsink_, TRUE);
    gst_app_sink_set_max_buffers(appsink_, 1);

    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        return false;
    }
    return true;
}

bool CameraCapture::grab(GpuFrame& frame)
{
    releaseGpuFrame(frame);

    GstSample* sample = gst_app_sink_try_pull_sample(appsink_, GST_MSECOND * 100);
    if (!sample) return false;

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return false;
    }

    auto now = std::chrono::steady_clock::now().time_since_epoch();
    frame.timestamp_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

    frame.width = width_;
    frame.height = height_;
    frame.sample = sample;

    bool ok = importNvmmToVpi(buffer, frame);
    if (!ok) {
        gst_sample_unref(sample);
        frame.sample = nullptr;
        return false;
    }

    return true;
}

bool CameraCapture::importNvmmToVpi(GstBuffer *buffer, GpuFrame &frame)
{
    NvBufSurface *surface = nullptr;

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        return false;
    }

    surface = reinterpret_cast<NvBufSurface *>(map.data);
    if (!surface) {
        gst_buffer_unmap(buffer, &map);
        return false;
    }

    VPIImageData data{};
    data.bufferType = VPI_IMAGE_BUFFER_NVBUFFER;
    data.buffer.fd = surface->surfaceList[0].bufferDesc;

    VPIImageWrapperParams params;
    VPIStatus st = vpiInitImageWrapperParams(&params);
    if (st != VPI_SUCCESS) {
        gst_buffer_unmap(buffer, &map);
        return false;
    }

    st = vpiImageCreateWrapper(&data, &params, VPI_BACKEND_CUDA | VPI_BACKEND_VIC, &frame.nv12);
    if (st != VPI_SUCCESS) {
        gst_buffer_unmap(buffer, &map);
        return false;
    }

    gst_buffer_unmap(buffer, &map);
    return true;
}
