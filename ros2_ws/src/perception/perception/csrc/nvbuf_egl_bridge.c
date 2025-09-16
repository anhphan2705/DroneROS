#include "gst/gst.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gstnvdsmeta.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>

EGLImageKHR gst_buffer_to_egl_image(GstBuffer *buf) {
    if (!buf) return NULL;

    // Retrieve the NvBufSurface pointer from the GstBuffer
    GstMapInfo in_map_info;
    if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ)) {
        g_printerr("Failed to map GstBuffer\n");
        return NULL;
    }

    NvBufSurface *surf = (NvBufSurface *) in_map_info.data;
    gst_buffer_unmap(buf, &in_map_info);

    if (!surf) {
        g_printerr("NvBufSurface not found in GstBuffer\n");
        return NULL;
    }

    if (NvBufSurfaceMapEglImage(surf, 0) != 0) {
        g_printerr("Failed to map NvBufSurface to EGLImage\n");
        return NULL;
    }

    return surf->surfaceList[0].mappedAddr.eglImage;
}


// gcc -fPIC -shared -o libnvbuf_egl_bridge.so nvbuf_egl_bridge.c \
    $(pkg-config --cflags --libs gstreamer-1.0) \
    -I/usr/src/jetson_multimedia_api/include \
    -I/opt/nvidia/deepstream/deepstream-7.0/sources/includes \
    -L/usr/lib/aarch64-linux-gnu/tegra \
    -lnvbufsurface -lnvbufsurftransform
