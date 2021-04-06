cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(LPRS)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/source)
include_directories(${PROJECT_SOURCE_DIR}/header)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package( OpenCV REQUIRED )

add_executable(LPRS source/main.cpp source/plate.cpp source/detector.cpp source/recognizer.cpp)
target_link_libraries(LPRS ${OpenCV_LIBS})
