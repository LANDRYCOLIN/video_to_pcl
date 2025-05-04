cmake_minimum_required(VERSION 3.8)
project(video_to_pcl)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找ROS2核心依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(ament_index_cpp REQUIRED)  # 添加这一行

# 关键修复：查找pcl_ros包并指定组件
find_package(pcl_ros REQUIRED COMPONENTS pcl_ros)

# OpenCV配置
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(video_to_pcl_node
  src/video_to_pcl_node.cpp
)

# 包含头文件目录
target_include_directories(video_to_pcl_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

# 自动链接ROS2依赖项
ament_target_dependencies(video_to_pcl_node
  rclcpp
  sensor_msgs
  cv_bridge
  pcl_conversions
  pcl_ros  # 直接依赖pcl_ros包
  ament_index_cpp  # 添加这一行
)

# 手动链接OpenCV和PCL基础库
target_link_libraries(video_to_pcl_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

# 安装可执行文件
install(TARGETS video_to_pcl_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装模型文件夹
install(DIRECTORY
  models/
  DESTINATION share/${PROJECT_NAME}/models/
)

# 导出ament依赖
ament_package()