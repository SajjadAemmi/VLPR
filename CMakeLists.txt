cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(pelak)

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package( OpenCV REQUIRED )

add_executable(pelak main.cpp)
target_link_libraries(pelak ${OpenCV_LIBS})
