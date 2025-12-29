#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <atomic>
#include <thread>
#include <string>
#include <cstring>   // memcpy

class CameraCaptureNode final : public rclcpp::Node
{
public:
    CameraCaptureNode()
    : Node("camera_capture_node",
            rclcpp::NodeOptions().use_intra_process_comms(true))
    {
        // ---- Params ----
        sensor_id_ = declare_parameter<int>("sensor_id", 0);
        width_     = declare_parameter<int>("width", 1920);
        height_    = declare_parameter<int>("height", 1080);
        fps_       = declare_parameter<int>("fps", 60);
        frame_id_  = declare_parameter<std::string>("frame_id", "camera");
        topic_     = declare_parameter<std::string>("topic", "/camera/image_nv12");

        // QoS: latest only, low latency
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.best_effort();
        qos.durability_volatile();

        pub_ = create_publisher<sensor_msgs::msg::Image>(topic_, qos);

        // Pre-size the message buffer once (NV12 = H*W*3/2)
        const size_t nv12_bytes = static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3 / 2;
        msg_ = std::make_unique<sensor_msgs::msg::Image>();
        msg_->header.frame_id = frame_id_;
        msg_->height = static_cast<uint32_t>(height_);
        msg_->width  = static_cast<uint32_t>(width_);
        msg_->encoding = "nv12";                 // not a REP-105 standard, but works if you control consumers
        msg_->is_bigendian = false;
        msg_->step = static_cast<uint32_t>(width_); // Y plane stride in bytes for tightly packed NV12
        msg_->data.resize(nv12_bytes);

        // ---- GStreamer init + pipeline ----
        gst_init(nullptr, nullptr);

        // IMPORTANT: we convert to CPU-visible NV12 for appsink mapping.
        // Later "Phase 2": keep NVMM and import via NvBufSurface to CUDA (true zero-copy).
        pipeline_str_ =
        "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id_) + " ! "
        "video/x-raw(memory:NVMM),width=" + std::to_string(width_) +
        ",height=" + std::to_string(height_) +
        ",framerate=" + std::to_string(fps_) + "/1,format=NV12 ! "
        "queue leaky=2 max-size-buffers=1 ! "
        "nvvidconv ! video/x-raw,format=NV12 ! "
        "appsink name=ros_sink max-buffers=1 drop=true sync=false";

        RCLCPP_INFO(get_logger(), "GStreamer pipeline:\n%s", pipeline_str_.c_str());

        GError *err = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str_.c_str(), &err);
        if (!pipeline_ || err) {
        std::string e = err ? err->message : "unknown";
        if (err) g_error_free(err);
        throw std::runtime_error("Failed to create pipeline: " + e);
        }

        GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline_), "ros_sink");
        appsink_ = GST_APP_SINK(sink);
        gst_object_unref(sink);

        // Make appsink “pull” mode (we’ll pull in our own thread)
        gst_app_sink_set_emit_signals(appsink_, FALSE);
        gst_app_sink_set_drop(appsink_, TRUE);
        gst_app_sink_set_max_buffers(appsink_, 1);

        // Start pipeline
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);

        // Start capture thread
        running_.store(true);
        capture_thread_ = std::thread(&CameraCaptureNode::captureLoop, this);
    }

    ~CameraCaptureNode() override
    {
        running_.store(false);
        if (capture_thread_.joinable()) capture_thread_.join();

        if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        appsink_ = nullptr;
        }
    }

private:
    void captureLoop()
    {
        // Pull newest sample; because drop=true and max-buffers=1, we get latest.
        while (running_.load() && rclcpp::ok()) {
        GstSample *sample = gst_app_sink_try_pull_sample(appsink_, 100000); // 100ms
        if (!sample) continue;

        GstBuffer *buffer = gst_sample_get_buffer(sample);
        GstCaps *caps = gst_sample_get_caps(sample);

        if (!buffer || !caps) {
            gst_sample_unref(sample);
            continue;
        }

        // Validate dimensions from caps (optional)
        GstStructure *s = gst_caps_get_structure(caps, 0);
        int w = 0, h = 0;
        gst_structure_get_int(s, "width", &w);
        gst_structure_get_int(s, "height", &h);

        if (w != width_ || h != height_) {
            // If caps mismatch, skip (or reallocate once if you want dynamic)
            gst_sample_unref(sample);
            continue;
        }

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            const size_t expected = static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3 / 2;
            if (map.size >= expected) {
            // Timestamp near capture (using ROS clock)
            msg_->header.stamp = this->get_clock()->now();

            // Copy NV12 bytes into preallocated buffer (single memcpy)
            std::memcpy(msg_->data.data(), map.data, expected);

            // Publish (intra-process helps; QoS keeps it lean)
            pub_->publish(*msg_);
            }
            gst_buffer_unmap(buffer, &map);
        }

        gst_sample_unref(sample);
        }
    }

    // Params
    int sensor_id_{0}, width_{1920}, height_{1080}, fps_{60};
    std::string frame_id_{"camera"};
    std::string topic_{"/camera/image_nv12"};

    // ROS
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    std::unique_ptr<sensor_msgs::msg::Image> msg_;

    // GStreamer
    std::string pipeline_str_;
    GstElement *pipeline_{nullptr};
    GstAppSink *appsink_{nullptr};

    // Thread
    std::atomic<bool> running_{false};
    std::thread capture_thread_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<CameraCaptureNode>();
        rclcpp::spin(node);
    } catch (const std::exception &e) {
        fprintf(stderr, "Fatal: %s\n", e.what());
    }
    rclcpp::shutdown();
    return 0;
}