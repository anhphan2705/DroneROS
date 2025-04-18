**OpenCV with both GPU (CUDA) and GStreamer support** on your **Jetson Orin Nano**:


Just run the `build_opencv_jetson.sh`


Or

---

### **1. Remove any pre-installed OpenCV**
```bash
sudo apt purge libopencv* python3-opencv
```

---

### **2. Install dependencies**
```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev
```

---

### **3. Clone OpenCV and OpenCV contrib**
```bash
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.8.0        # or a known compatible version
cd ../opencv_contrib
git checkout 4.8.0
```

---

### **4. Build OpenCV with CUDA + GStreamer**
```bash
cd ~/opencv
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_GSTREAMER_0_10=OFF \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON3_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
      -D PYTHON3_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
      ..
```

Make sure to verify the Python paths are correct.

---

### **5. Compile (takes time)**
```bash
make -j$(nproc)
```

If it crashes due to RAM, use:
```bash
make -j2
```

---

### **6. Install**
```bash
sudo make install
sudo ldconfig
```

---

### **7. Verify**
```bash
python3 -c "import cv2; print(cv2.getBuildInformation())"
```
Look for:
- `GStreamer: YES`
- `CUDA: YES`
- `cuDNN: YES`
- `Python 3: YES` path to correct interpreter

---