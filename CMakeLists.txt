cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(demo)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(demo demo.cpp)
target_link_libraries(demo ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET demo PROPERTY CXX_STANDARD 14)