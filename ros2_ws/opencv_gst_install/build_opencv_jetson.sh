#!/bin/bash

# Set up paths
OPENCV_DIR=~/IDS/opencv
OPENCV_CONTRIB_DIR=~/IDS/opencv_contrib
BUILD_DIR=$OPENCV_DIR/build

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit

# Run CMake with Jetson-specific optimizations
cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
-D OPENCV_EXTRA_MODULES_PATH=$OPENCV_CONTRIB_DIR/modules \
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
-D WITH_QT=OFF \
-D WITH_LAPACK=ON \
-D WITH_GTK=ON \
-D WITH_NVCUVID=ON \
-D WITH_NVCUVENC=ON \
-D BUILD_opencv_sfm=ON \
-D BUILD_opencv_python3=ON \
-D BUILD_EXAMPLES=ON ..

# Build and install
make -j$(nproc)
sudo make install
sudo ldconfig

echo "✅ OpenCV build and installation complete!"
