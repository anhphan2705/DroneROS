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

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED
  gstreamer-1.0
  gstreamer-app-1.0
)

# ============================
# Executable
# ============================
add_executable(camera_capture_node
  src/camera_capture_node.cpp
)

target_include_directories(camera_capture_node PRIVATE
  ${GST_INCLUDE_DIRS}
)

target_link_libraries(camera_capture_node
  ${GST_LIBRARIES}
)

ament_target_dependencies(camera_capture_node
  rclcpp
  sensor_msgs
)

# ============================
# Install
# ============================
install(TARGETS camera_capture_node
  DESTINATION lib/${PROJECT_NAME}
)

# ============================
# Disable tests for RT builds
# ============================
if(BUILD_TESTING)
  message(STATUS "BUILD_TESTING enabled, but disabled for RT node")
endif()

ament_package()
