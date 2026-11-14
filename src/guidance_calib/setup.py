from setuptools import find_packages, setup

package_name = 'guidance_calib'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='todo@example.com',
    description='Overhead camera calibration: intrinsics and image-to-ground-plane mapping.',
    license='TODO',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'calib_node = guidance_calib.calib_node:main',
            'generate_markers = guidance_calib.generate_markers:main',
        ],
    },
)
