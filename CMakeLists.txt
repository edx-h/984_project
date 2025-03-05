cmake_minimum_required(VERSION 3.5.0)
project(cuda VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(SRC_PATH ${PROJECT_SOURCE_DIR}/comp)
file(GLOB_RECURSE SRC_FILES "${SRC_PATH}/*.cu")
# 打印调试信息
message(STATUS "Source files: ${SRC_FILES}")
add_executable(cuda "${SRC_PATH}/algorithm_executable.cu")
include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
