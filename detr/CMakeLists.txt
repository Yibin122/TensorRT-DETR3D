cmake_minimum_required(VERSION 2.8.0)
project(detr_trt)
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_library(NVINFER NAMES nvinfer)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ~/TensorRT-8.0.3.4/include
)

add_executable(${PROJECT_NAME}
  detr_test.cc
  detr.cc
)

target_link_libraries(${PROJECT_NAME}
  # ${NVINFER}
  ~/TensorRT-8.0.3.4/lib/libnvinfer.so
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
