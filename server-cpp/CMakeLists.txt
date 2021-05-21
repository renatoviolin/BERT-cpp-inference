set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

include_directories("/opt/X11/include")
include_directories("/usr/local/include/")

set(Torch_DIR "/Users/renato/Documents/dev/libtorch/share/cmake/Torch")
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(bert-cpp)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(bert-cpp bert-server.cpp libs/infer.cpp)
# add_executable(bert-cpp bert.cpp)

target_link_libraries(bert-cpp "${TORCH_LIBRARIES}")
target_link_libraries(bert-cpp "${OpenCV_LIBS}")
target_link_libraries(bert-cpp ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

set_property(TARGET bert-cpp PROPERTY CXX_STANDARD 14)
set_property(TARGET bert-cpp PROPERTY OUTPUT_NAME bert-cpp)