#!/bin/bash

set -e  # Exit on any error

echo "📦 Installing OpenCV build dependencies..."

# Basic build tools and Python
sudo apt update
sudo apt install -y \
    build-essential cmake git pkg-config \
    python3 python3-dev python3-pip python3-numpy \
    libpython3-dev

# GUI and media I/O
sudo apt install -y \
    libgtk-3-dev qtbase5-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev v4l-utils \
    libxvidcore-dev libx264-dev \
    gstreamer1.0-tools libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Math and acceleration
sudo apt install -y \
    liblapack-dev libblas-dev libatlas-base-dev libeigen3-dev

# Optional modules (viz, sfm, etc.)
sudo apt install -y \
    libvtk7-dev libgflags-dev libgoogle-glog-dev \
    libprotobuf-dev protobuf-compiler \
    libgphoto2-dev libopenni2-dev

echo "✅ Dependencies installed."

# Get absolute path to the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Set up OpenCV source paths relative to this script
OPENCV_DIR="$SCRIPT_DIR/opencv"
OPENCV_CONTRIB_DIR="$SCRIPT_DIR/opencv_contrib"
BUILD_DIR="$OPENCV_DIR/build"

# Clean and create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit

echo "🔧 Running CMake configuration..."

cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
-D OPENCV_EXTRA_MODULES_PATH="$OPENCV_CONTRIB_DIR/modules" \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D WITH_GSTREAMER=ON \
-D WITH_LIBV4L=ON \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=7.2 \
-D WITH_TBB=ON \
-D WITH_OPENGL=ON \
-D WITH_QT=ON \
-D WITH_GTK=ON \
-D WITH_LAPACK=ON \
-D LAPACK_LIBRARIES="/usr/lib/aarch64-linux-gnu/liblapack.so;/usr/lib/aarch64-linux-gnu/libblas.so" \
-D WITH_EIGEN=ON \
-D WITH_NVCUVID=ON \
-D WITH_NVCUVENC=ON \
-D BUILD_opencv_sfm=ON \
-D BUILD_opencv_viz=ON \
-D BUILD_opencv_python3=ON \
-D BUILD_EXAMPLES=ON ..

echo "🚀 Building OpenCV..."
make -j$(nproc)

echo "📥 Installing OpenCV..."
sudo make install
sudo ldconfig

echo "✅ OpenCV build and installation complete!"

echo "🧪 Verifying OpenCV installation..."
python3 -c "import cv2; print(cv2.getBuildInformation())"
