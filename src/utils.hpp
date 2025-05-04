#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>

class DepthEstimator {
private:
    cv::dnn::Net net;
    const int input_width = 256;
    const int input_height = 256;
    bool model_loaded = false;

public:
    // 默认构造函数
    DepthEstimator() {}

    // 初始化方法，接受模型路径
    bool initialize(const std::string& model_path) {
        try {
            // 尝试加载MiDaS模型
            net = cv::dnn::readNet(model_path);
            
            // 将计算设置为CPU（或根据需求设置为GPU）
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            
            model_loaded = true;
            std::cout << "成功加载深度估计模型: " << model_path << std::endl;
            return true;
        }
        catch (const cv::Exception& e) {
            std::cerr << "无法加载深度估计模型: " << e.what() << std::endl;
            std::cerr << "将使用备用深度生成方法" << std::endl;
            return false;
        }
    }

    void estimate(const cv::Mat& rgb, cv::Mat& depth) {
        if (!model_loaded) {
            // 如果模型加载失败，使用备用方法
            generateFallbackDepth(rgb, depth);
            return;
        }
        
        // 预处理图像
        cv::Mat input;
        cv::cvtColor(rgb, input, cv::COLOR_BGR2RGB);
        
        // 调整大小并归一化
        cv::Mat blob;
        cv::dnn::blobFromImage(input, blob, 1.0/255.0, 
                              cv::Size(input_width, input_height), 
                              cv::Scalar(0.485, 0.456, 0.406), 
                              true, false);
        
        // 通过网络前向传播
        net.setInput(blob);
        cv::Mat output = net.forward();
        
        // 调整深度图大小以匹配原始图像
        cv::Mat resized_depth;
        cv::resize(output.reshape(1, input_height), resized_depth, rgb.size());
        
        // 归一化深度值并反转（MiDaS输出的是相对深度，值越大距离越远）
        double min_val, max_val;
        cv::minMaxLoc(resized_depth, &min_val, &max_val);
        
        // 转换为正常的深度值（越大越远）
        depth = (resized_depth - min_val) / (max_val - min_val);
        
        // 将相对深度转换为更真实的深度范围（例如1-10米）
        depth.convertTo(depth, CV_32F, 9.0, 1.0); // 1-10米范围
    }

private:
    // 备用方法，如果模型加载失败则使用
    void generateFallbackDepth(const cv::Mat& rgb, cv::Mat& depth) {
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
};

// 全局单例深度估计器
static DepthEstimator* global_estimator = nullptr;
static std::string global_model_path = "";

// 设置模型路径的函数
inline void set_model_path(const std::string& path) {
    global_model_path = path;
}

// 主函数，替换原来的generate_sample_depth
inline void generate_sample_depth(const cv::Mat& rgb, cv::Mat& depth) {
    // 懒加载单例模式
    if (global_estimator == nullptr) {
        global_estimator = new DepthEstimator();
        if (!global_model_path.empty()) {
            global_estimator->initialize(global_model_path);
        } else {
            std::cerr << "警告：未设置模型路径，将使用备用深度生成方法" << std::endl;
        }
    }
    global_estimator->estimate(rgb, depth);
}