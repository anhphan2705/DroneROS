#include <rclcpp/rclcpp.hpp>

#include "perc/camera/camera_capture.hpp"
#include "perc/common/gpu_frame.hpp"

class PerceptionGpuNode : public rclcpp::Node
{
public:
    PerceptionGpuNode()
    : Node("perception_gpu_node")
    {
        int sensor = declare_parameter("sensor_id", 0);
        int width  = declare_parameter("width", 1920);
        int height = declare_parameter("height", 1080);
        int fps    = declare_parameter("fps", 60);

        if (!camera_.init(sensor, width, height, fps)) {
            throw std::runtime_error("Camera init failed");
        }

        worker_ = std::thread(&PerceptionGpuNode::loop, this);
    }

    ~PerceptionGpuNode()
    {
        running_ = false;
        if (worker_.joinable()) worker_.join();
    }

private:
    void loop()
    {
        GpuFrame frame;

        while (rclcpp::ok() && running_) {
            if (!camera_.grab(frame)) continue;

            // 🔜 Rectifier(frame)
            // 🔜 StereoDepth(frame)
            // 🔜 Detector(frame)

            // For now: just prove GPU capture works
        }
    }

    CameraCapture camera_;
    std::thread worker_;
    std::atomic<bool> running_{true};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PerceptionGpuNode>());
    rclcpp::shutdown();
    return 0;
}