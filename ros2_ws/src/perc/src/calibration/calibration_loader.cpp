#include "perc/calibration/calibration_loader.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>

bool CalibrationLoader::readSide(cv::FileStorage& fs,
                                 const std::string& prefix,
                                 cv::Mat& map_x,
                                 cv::Mat& map_y)
{
    cv::Mat m1 = fs[prefix + "_x"].mat();
    cv::Mat m2 = fs[prefix + "_y"].mat();

    if (m1.empty() || m2.empty()) {
        return false;
    }

    if (m1.type() == CV_16SC2) {
        cv::Mat map2f;
        cv::convertMaps(m1, m2, map_x, map_y, CV_32FC1);
        return !map_x.empty() && !map_y.empty();
    }

    if (m1.type() == CV_32FC1 && m2.type() == CV_32FC1) {
        map_x = m1.clone();
        map_y = m2.clone();
        return true;
    }

    if (m1.type() == CV_32FC2 && m1.channels() == 2) {
        std::vector<cv::Mat> ch;
        cv::split(m1, ch);
        if (ch.size() != 2) {
            return false;
        }
        map_x = ch[0].clone();
        map_y = ch[1].clone();
        return true;
    }

    return false;
}

bool CalibrationLoader::loadStereoPair(const std::string& file, StereoPairCalibration& out)
{
    cv::FileStorage fs(file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }

    if (!readSide(fs, "stereo_map_left", out.left_map_x, out.left_map_y)) {
        return false;
    }
    if (!readSide(fs, "stereo_map_right", out.right_map_x, out.right_map_y)) {
        return false;
    }

    cv::Mat M1 = fs["camera_matrix_left"].mat();
    cv::Mat T  = fs["translation_vector"].mat();

    if (M1.empty() || T.empty()) {
        return false;
    }

    if (M1.type() == CV_64F) {
        out.fx = static_cast<float>(M1.at<double>(0, 0));
    } else {
        out.fx = M1.at<float>(0, 0);
    }

    double tx = 0.0;
    if (T.type() == CV_64F) {
        tx = T.at<double>(0, 0);
    } else {
        tx = T.at<float>(0, 0);
    }

    out.baseline_m = static_cast<float>(std::abs(tx) / 1000.0);

    return true;
}

bool CalibrationLoader::loadDualStereo(const std::string& file0,
                                       const std::string& file1,
                                       DualStereoCalibration& out)
{
    return loadStereoPair(file0, out.pair0) &&
           loadStereoPair(file1, out.pair1);
}