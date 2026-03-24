#pragma once

#include <opencv2/core.hpp>

struct StereoPairCalibration
{
    cv::Mat left_map_x;
    cv::Mat left_map_y;
    cv::Mat right_map_x;
    cv::Mat right_map_y;

    float fx{0.0f};
    float baseline_m{0.0f};
};

struct DualStereoCalibration
{
    StereoPairCalibration pair0;
    StereoPairCalibration pair1;
};