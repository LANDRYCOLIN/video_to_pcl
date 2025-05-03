#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "utils.hpp"

class VideoToPclNode : public rclcpp::Node {
public:
    VideoToPclNode() : Node("video_to_pcl_node") {
        declare_parameter("video_path", "input.mp4");
        declare_parameter("output_cloud", "output.pcd");
        RCLCPP_INFO(get_logger(), "节点初始化完成");
        process_video();
    }

private:
    void process_video() {
        std::string video_path = get_parameter("video_path").as_string();
        cv::VideoCapture cap(video_path);
        
        if (!cap.isOpened()) {
            RCLCPP_ERROR(get_logger(), "视频打开失败: %s", video_path.c_str());
            return;
        }

        // 获取视频参数
        const int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        RCLCPP_INFO(get_logger(), "视频分辨率: %dx%d", width, height);

        // 动态计算内参
        const float fx = width / (2 * tan(60.0 * CV_PI / 360));
        const float fy = fx;
        const float cx = width / 2.0;
        const float cy = height / 2.0;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cv::Mat frame, depth;
        int frame_count = 0;

        while (cap.read(frame) && frame_count < 100) { // 限制处理帧数为前 100 帧
            RCLCPP_INFO(get_logger(), "处理第 %d 帧", ++frame_count);
            
            // 生成深度图
            generate_sample_depth(frame, depth);
            
            // 生成点云
            for (int y = 0; y < depth.rows; y += 2) {
                for (int x = 0; x < depth.cols; x += 2) {
                    float z = depth.at<float>(y, x);
                    if (z > 0) {
                        pcl::PointXYZRGB point;
                        point.x = (x - cx) * z / fx;
                        point.y = (y - cy) * z / fy;
                        point.z = z;
                        point.r = frame.at<cv::Vec3b>(y, x)[2];
                        point.g = frame.at<cv::Vec3b>(y, x)[1];
                        point.b = frame.at<cv::Vec3b>(y, x)[0];
                        cloud->push_back(point);
                    }
                }
            }
            
            RCLCPP_INFO(get_logger(), "当前点云点数: %zu", cloud->size());
        }

        // 保存点云
        if (!cloud->empty()) {
            pcl::io::savePCDFile(get_parameter("output_cloud").as_string(), *cloud);
            RCLCPP_INFO(get_logger(), "点云已保存至: %s", get_parameter("output_cloud").as_string().c_str());
        } else {
            RCLCPP_ERROR(get_logger(), "无法保存空点云！");
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoToPclNode>());
    rclcpp::shutdown();
    return 0;
}