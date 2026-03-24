#pragma once

#include "perc/common/calibration_data.hpp"
#include <string>

class CalibrationLoader
{
public:
    static bool loadStereoPair(const std::string& file, StereoPairCalibration& out);
    static bool loadDualStereo(const std::string& file0,
                               const std::string& file1,
                               DualStereoCalibration& out);

private:
    static bool readSide(cv::FileStorage& fs,
                         const std::string& prefix,
                         cv::Mat& map_x,
                         cv::Mat& map_y);
};