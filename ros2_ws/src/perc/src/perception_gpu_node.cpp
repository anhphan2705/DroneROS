#include <rclcpp/rclcpp.hpp>
#include <filesystem>

#include "perc/camera/camera_capture.hpp"
#include "perc/pipeline/stereo_pipeline.hpp"
#include "perc/common/gpu_frame.hpp"
#include "perc/calibration/calibration_loader.hpp"

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
        std::string calib0 = declare_parameter<std::string>("calibration_file_0", "");
        std::string calib1 = declare_parameter<std::string>("calibration_file_1", "");

        if (!std::filesystem::exists(calib0) || !std::filesystem::exists(calib1)) {
            throw std::runtime_error("Calibration files not found");
        }

        RCLCPP_INFO(get_logger(), "Using calibration files: %s, %s", calib0.c_str(), calib1.c_str());

        DualStereoCalibration calib;
        if (!CalibrationLoader::loadDualStereo(calib0, calib1, calib)) {
            throw std::runtime_error("Failed to load calibration files");
        }

        if (!camera_.init(sensor, width, height, fps)) {
            throw std::runtime_error("Camera init failed");
        }

        if (!pipeline_.init(width, height, calib)) {
            throw std::runtime_error("Pipeline init failed");
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
            if (!camera_.grab(frame)) {
                continue;
            }

            if (!pipeline_.process(frame)) {
                RCLCPP_WARN_THROTTLE(
                    get_logger(),
                    *get_clock(),
                    2000,
                    "Pipeline process failed");
                releaseGpuFrame(frame);
                continue;
            }

            RCLCPP_INFO_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Frame grabbed and processed on GPU: %dx%d ts=%lu",
                frame.width,
                frame.height,
                static_cast<unsigned long>(frame.timestamp_ns));

            releaseGpuFrame(frame);
        }
    }

    CameraCapture camera_;
    StereoPipeline pipeline_;
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