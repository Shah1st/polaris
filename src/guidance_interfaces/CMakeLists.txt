cmake_minimum_required(VERSION 3.8)
project(guidance_interfaces)

find_package(ament_cmake REQUIRED)

# TODO(interfaces): add custom msg/srv/action files once the data flow between nodes is settled
# (e.g. mission state, calibration result), then uncomment and list them:
#
# find_package(rosidl_default_generators REQUIRED)
# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/Example.msg"
# )
# ament_export_dependencies(rosidl_default_runtime)

ament_package()
