perception/
├── CMakeLists.txt
├── package.xml
│
├── launch/
│   └── perception_gpu.launch.py
│
├── include/perception_gpu/
│   ├── perception_gpu_node.hpp
│   │
│   ├── camera/
│   │   ├── camera_capture.hpp
│   │
│   ├── rectification/
│   │   ├── rectifier.hpp
│   │   └── calibration_loader.hpp
│   │
│   ├── depth/
│   │   └── stereo_depth.hpp
│   │
│   ├── detection/
│   │   ├── trt_detector.hpp
│   │
│   ├── tracking/
│   │   └── bytetrack.hpp
│   │
│   ├── fusion/
│   │   └── speed_estimator.hpp
│   │
│   └── ros/
│       └── ros_interface.hpp
│
├── src/
│   ├── perception_gpu_node.cpp   ← main()
│
│   ├── camera/
│   │   └── camera_capture.cpp
│
│   ├── rectification/
│   │   └── rectifier.cpp
│
│   ├── depth/
│   │   └── stereo_depth.cpp
│
│   ├── detection/
│   │   └── trt_detector.cpp
│
│   ├── tracking/
│   │   └── bytetrack.cpp
│
│   ├── fusion/
│   │   └── speed_estimator.cpp
│
│   └── ros/
│       └── ros_interface.cpp
│
└── resources/
    ├── calibration/
    └── models/
