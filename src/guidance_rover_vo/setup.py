from setuptools import find_packages, setup

package_name = 'guidance_rover_vo'

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
    description='Bridge from the rover monocular mapper: visual odometry and local 3D points.',
    license='TODO',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'rover_vo_bridge_node = guidance_rover_vo.rover_vo_bridge_node:main',
        ],
    },
)
