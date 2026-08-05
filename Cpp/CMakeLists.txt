cmake_minimum_required(VERSION 3.15)
project(DroneController)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MAVSDK REQUIRED)

# Add your C++ source files
add_executable(drone_controller src/TestFlight.cpp)

target_link_libraries(drone_controller
    MAVSDK::mavsdk
)
