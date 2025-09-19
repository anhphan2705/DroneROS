from setuptools import find_packages, setup
import os
import glob

package_name = 'perception'
model_files = glob.glob('resource/models/*')
calibrated_params_dir = 'perception/calibrated_params'
calibration_files = glob.glob(os.path.join(calibrated_params_dir, '*.yml')) if os.path.exists(calibrated_params_dir) else []

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=['perception', 'perception.*'], exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), [
            'launch/camera_launch.py',
            'launch/visualizer_launch.py',
        ]),
        (os.path.join('share', package_name, 'models'), model_files),
        (os.path.join('share', package_name, 'calibrated_params'), calibration_files),
    ],
    install_requires=['setuptools', 'rosidl_runtime_py'],
    zip_safe=True,
    maintainer='anhphan',
    maintainer_email='anhphanvt2705@gmail.com',
    description='Manage all perception devices in the RP-1 Base Station',
    license='Private',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_gpu_node = perception.camera_gpu_node:main',
            'autofocus_node = perception.autofocus_node:main',
            'manual_focus_node = perception.manual_focus_node:main',
            'sync_capture_node = perception.sync_capture_node:main',
            'object_detection_node = perception.object_detection_node:main',
            'byte_track_node = perception.byte_track_node:main',
            'tracked_bbox_overlay_node = perception.tracked_bbox_overlay_node:main',
            'object_depth_fusion_node = perception.object_depth_fusion_node:main',
            'classification_node = perception.classification_node:main',
            'speed_estimator_node = perception.speed_estimator_node:main',
            'id_mapper_node = perception.id_mapper_node:main',
            'bag2mp4 = perception.bag2mp4:main',
        ],
    },
)
