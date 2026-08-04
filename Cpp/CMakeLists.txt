cmake_minimum_required(VERSION 3.15)
project(DroneController)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# Add your C++ source files
add_executable(balloon_detect src/Balloon_detection.cpp)
target_link_libraries(balloon_detect ${OpenCV_LIBS})
