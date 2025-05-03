#pragma once
#include <opencv2/opencv.hpp>

inline void generate_sample_depth(const cv::Mat& rgb, cv::Mat& depth) {
    depth.create(rgb.size(), CV_32F);
    const int center_x = rgb.cols / 2;
    const int center_y = rgb.rows / 2;
    
    for (int y = 0; y < rgb.rows; y++) {
        for (int x = 0; x < rgb.cols; x++) {
            // 生成梯度深度（中心深，边缘浅）
            float dist = sqrt(pow(x - center_x, 2) + pow(y - center_y, 2));
            depth.at<float>(y, x) = 1.0 + dist * 0.01; // 确保深度>0
        }
    }
}