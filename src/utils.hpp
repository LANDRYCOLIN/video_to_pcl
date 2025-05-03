#pragma once
#include <opencv2/opencv.hpp>

inline void generate_sample_depth(const cv::Mat& rgb, cv::Mat& depth) {
    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    
    // 检测边缘
    cv::Mat edges;
    cv::Canny(gray, edges, 100, 200);
    
    // 使用高斯模糊扩散边缘
    cv::Mat edge_depth;
    cv::GaussianBlur(edges, edge_depth, cv::Size(21, 21), 5, 5);
    
    // 基本深度由亮度值决定
    cv::Mat brightness;
    cv::cvtColor(rgb, brightness, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(brightness, channels);
    
    // 基于亮度和边缘的深度估计
    depth.create(rgb.size(), CV_32F);
    for (int y = 0; y < rgb.rows; y++) {
        for (int x = 0; x < rgb.cols; x++) {
            float edge_value = edge_depth.at<uchar>(y, x) / 255.0f;
            float bright_value = channels[2].at<uchar>(y, x) / 255.0f;
            
            // 边缘通常是前景，较亮的区域也可能更接近
            float depth_value = 2.0f + 8.0f * (1.0f - edge_value * 0.6f - bright_value * 0.4f);
            depth.at<float>(y, x) = depth_value;
        }
    }
}