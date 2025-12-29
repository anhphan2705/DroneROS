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
    : Node(
        "camera_capture_node",
        rclcpp::NodeOptions().use_intra_process_comms(true))
    {
        // ---------------- Parameters ----------------
        sensor_id_ = declare_parameter<int>("sensor_id", 0);
        width_     = declare_parameter<int>("width", 1920);
        height_    = declare_parameter<int>("height", 1080);
        fps_       = declare_parameter<int>("fps", 60);
        frame_id_  = declare_parameter<std::string>("frame_id", "camera");
        topic_     = declare_parameter<std::string>("topic", "/camera/image_nv12");
        debug_rgb_ = declare_parameter<bool>("debug_rgb", false);

        // ---------------- QoS ----------------
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.best_effort();
        qos.durability_volatile();

        pub_ = create_publisher<sensor_msgs::msg::Image>(topic_, qos);

        // ---------------- NV12 message (preallocated) ----------------
        const size_t nv12_bytes =
        static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3 / 2;

        msg_ = std::make_unique<sensor_msgs::msg::Image>();
        msg_->header.frame_id = frame_id_;
        msg_->height = height_;
        msg_->width  = width_;
        msg_->encoding = "nv12";
        msg_->is_bigendian = false;
        msg_->step = width_;
        msg_->data.resize(nv12_bytes);

        // ---------------- Debug RGB publisher ----------------
        if (debug_rgb_) {
            debug_pub_ = create_publisher<sensor_msgs::msg::Image>(
                "/camera/image_rgb8", qos);

            debug_msg_.header.frame_id = frame_id_;
            debug_msg_.height = height_;
            debug_msg_.width  = width_;
            debug_msg_.encoding = "rgb8";
            debug_msg_.is_bigendian = false;
            debug_msg_.step = width_ * 3;
            debug_msg_.data.resize(width_ * height_ * 3);
        }

        // ---------------- GStreamer ----------------
        gst_init(nullptr, nullptr);

        if (debug_rgb_) {
            pipeline_str_ =
                "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id_) + " ! "
                "video/x-raw(memory:NVMM),width=" + std::to_string(width_) +
                ",height=" + std::to_string(height_) +
                ",framerate=" + std::to_string(fps_) + "/1,format=NV12 ! "
                "tee name=t "
                "t. ! queue leaky=2 max-size-buffers=1 ! "
                "nvvidconv ! video/x-raw,format=NV12 ! "
                "appsink name=nv12_sink max-buffers=1 drop=true sync=false "
                "t. ! queue leaky=2 max-size-buffers=1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "appsink name=rgb_sink max-buffers=1 drop=true sync=false";
        } else {
            pipeline_str_ =
                "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id_) + " ! "
                "video/x-raw(memory:NVMM),width=" + std::to_string(width_) +
                ",height=" + std::to_string(height_) +
                ",framerate=" + std::to_string(fps_) + "/1,format=NV12 ! "
                "queue leaky=2 max-size-buffers=1 ! "
                "nvvidconv ! video/x-raw,format=NV12 ! "
                "appsink name=ros_sink max-buffers=1 drop=true sync=false";
        }

        RCLCPP_INFO(get_logger(), "GStreamer pipeline:\n%s", pipeline_str_.c_str());

        GError *err = nullptr;
        pipeline_ = gst_parse_launch(pipeline_str_.c_str(), &err);
        if (!pipeline_ || err) {
            std::string e = err ? err->message : "unknown";
            if (err) g_error_free(err);
            throw std::runtime_error("Failed to create pipeline: " + e);
        }

        // ---------------- Get appsinks ----------------
        if (debug_rgb_) {
            GstElement *nv12 = gst_bin_get_by_name(GST_BIN(pipeline_), "nv12_sink");
            GstElement *rgb  = gst_bin_get_by_name(GST_BIN(pipeline_), "rgb_sink");

            nv12_appsink_ = GST_APP_SINK(nv12);
            rgb_appsink_  = GST_APP_SINK(rgb);

            gst_object_unref(nv12);
            gst_object_unref(rgb);
        } else {
            GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline_), "ros_sink");
            nv12_appsink_ = GST_APP_SINK(sink);
            gst_object_unref(sink);
        }

        gst_element_set_state(pipeline_, GST_STATE_PLAYING);

        // ---------------- Thread ----------------
        running_.store(true);
        capture_thread_ = std::thread(&CameraCaptureNode::captureLoop, this);
    }

    ~CameraCaptureNode() override
    {
        running_.store(false);
        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }

        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }
    }

private:
    void captureLoop()
    {
        while (running_.load() && rclcpp::ok()) {
            // -------- NV12 --------
            GstSample *nv12_sample =
                gst_app_sink_try_pull_sample(nv12_appsink_, 100000);
            if (!nv12_sample) continue;

            GstBuffer *nv12_buf = gst_sample_get_buffer(nv12_sample);
            GstMapInfo map;

            if (nv12_buf && gst_buffer_map(nv12_buf, &map, GST_MAP_READ)) {
                const size_t expected =
                static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3 / 2;

                if (map.size >= expected) {
                    msg_->header.stamp = get_clock()->now();
                    std::memcpy(msg_->data.data(), map.data, expected);
                    pub_->publish(*msg_);
                }
                gst_buffer_unmap(nv12_buf, &map);
            }
            gst_sample_unref(nv12_sample);

            // -------- RGB DEBUG --------
            if (debug_rgb_) {
                GstSample *rgb_sample =
                    gst_app_sink_try_pull_sample(rgb_appsink_, 0);

                if (rgb_sample) {
                        GstBuffer *rgb_buf = gst_sample_get_buffer(rgb_sample);
                        GstMapInfo rgb_map;

                        if (rgb_buf && gst_buffer_map(rgb_buf, &rgb_map, GST_MAP_READ)) {
                            const size_t pixels = static_cast<size_t>(width_) * static_cast<size_t>(height_);

                            // BGRx → RGB (drop alpha channel)
                            uint8_t *dst = debug_msg_.data.data();
                            uint8_t *src = rgb_map.data;

                            for (size_t i = 0, j = 0; i < pixels; ++i) {
                                dst[j++] = src[i * 4 + 2]; // R
                                dst[j++] = src[i * 4 + 1]; // G
                                dst[j++] = src[i * 4 + 0]; // B
                        }

                        debug_msg_.header.stamp = msg_->header.stamp;
                        debug_pub_->publish(debug_msg_);

                        gst_buffer_unmap(rgb_buf, &rgb_map);
                    }

                    gst_sample_unref(rgb_sample);
                }
            }
        }
    }

    // ---------------- Members ----------------
    int sensor_id_{0}, width_{1920}, height_{1080}, fps_{60};
    bool debug_rgb_{false};
    std::string frame_id_;
    std::string topic_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    std::unique_ptr<sensor_msgs::msg::Image> msg_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    sensor_msgs::msg::Image debug_msg_;

    GstElement *pipeline_{nullptr};
    GstAppSink *nv12_appsink_{nullptr};
    GstAppSink *rgb_appsink_{nullptr};
    std::string pipeline_str_;

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