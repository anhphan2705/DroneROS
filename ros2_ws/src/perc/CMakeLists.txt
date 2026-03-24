cmake_minimum_required(VERSION 3.8)
project(perc)

# ============================
# Compiler + RT optimization
# ============================
add_compile_options(
  -O3
  -march=native
  -ffast-math
)

# ============================
# Dependencies
# ============================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED
  gstreamer-1.0
  gstreamer-app-1.0
)

# ============================
# Executable
# ============================

# camera debug node
add_executable(camera_capture_node
  src/camera_capture_node.cpp
)

target_link_libraries(camera_capture_node
  ${GST_LIBRARIES}
)

ament_target_dependencies(camera_capture_node
  rclcpp
  sensor_msgs
)

# main perception node
add_executable(perception_gpu_node
  src/perception_gpu_node.cpp
  src/camera/camera_capture.cpp
  src/pipeline/stereo_pipeline.cpp
  src/calibration/calibration_loader.cpp
)

target_link_libraries(perception_gpu_node
  ${GST_LIBRARIES}
  ${OpenCV_LIBRARIES}
  EGL
  nvvpi
)

ament_target_dependencies(perception_gpu_node
  rclcpp
  sensor_msgs
)

# ============================
# Common include dirs
# ============================
include_directories(
  include
  ${GST_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  /usr/src/jetson_multimedia_api/include
)

# ============================
# Install
# ============================
install(TARGETS 
  camera_capture_node
  perception_gpu_node
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
install(
  DIRECTORY calibration
  DESTINATION share/${PROJECT_NAME}
)

# ============================
# Disable tests for RT builds
# ============================
if(BUILD_TESTING)
  message(STATUS "BUILD_TESTING enabled, but disabled for RT node")
endif()

ament_package()
