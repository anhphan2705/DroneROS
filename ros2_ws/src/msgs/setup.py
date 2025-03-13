from setuptools import find_packages, setup

package_name = 'msgs'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anhphan',
    maintainer_email='anhphanvt2705@gmail.com',
    description='Manage all internal messages in BaseROS',
    license='Private',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
