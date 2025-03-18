import os
import glob
from setuptools import find_packages, setup

package_name = 'calibration'

calibrated_params_dir = 'calibration/calibrated_params'
calibration_files = glob.glob(os.path.join(calibrated_params_dir, '*.yml')) if os.path.exists(calibrated_params_dir) else []

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['calibration', 'calibration.*'], exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.yaml')),
        (os.path.join('share', package_name, 'calibrated_params'), calibration_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anhphan',
    maintainer_email='anhphanvt2705@gmail.com',
    description='Calibration package fo calibrate sensors in base station',
    license='Private',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stereo_calibration_node = calibration.stereo_calibration_node:main',
            'camera_rectification_node = calibration.camera_rectification_node:main',
        ],
    },
)
