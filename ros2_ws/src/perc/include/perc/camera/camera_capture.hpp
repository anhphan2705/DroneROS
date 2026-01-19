#pragma once

#include <string>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <EGL/egl.h>
#include <vpi/VPI.h>

#include "perc/common/gpu_frame.hpp"

class CameraCapture
{
public:
    CameraCapture();
    ~CameraCapture();

    bool init(int sensor_id, int width, int height, int fps);
    bool grab(GpuFrame& frame);

private:
    bool initGStreamer();
    bool importNvmmToVpi(GstBuffer* buffer, GpuFrame& frame);

    int sensor_id_{0};
    int width_{0}, height_{0}, fps_{0};

    GstElement* pipeline_{nullptr};
    GstAppSink* appsink_{nullptr};

    EGLDisplay egl_display_{EGL_NO_DISPLAY};
};