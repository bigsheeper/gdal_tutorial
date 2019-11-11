cmake_minimum_required(VERSION 3.5)

project(gdal_tutorial LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
message("CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC -D_FORCE_INLINES -arch sm_60 --expt-extended-lambda")

set(CUDA_INCLUDE /usr/local/cuda/include)
set(CUDA_LIB /usr/local/cuda/lib64)

add_executable(gdal_org tutorial_1/gdal_org_read_and_write.cpp)
add_executable(translate tutorial_2/translate.cpp)
add_executable(polygon_op tutorial_3/polygon_op.cpp)
add_executable(gdal_opengl tutorial_4/gdal_opengl.cpp)

target_link_libraries(gdal_org gdal)
target_link_libraries(translate gdal)
target_link_libraries(polygon_op gdal)
target_link_libraries(gdal_opengl gdal GL EGL)

add_subdirectory(tutorial_5)