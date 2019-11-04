cmake_minimum_required(VERSION 3.14)
project(gdal_test)

set(CMAKE_CXX_STANDARD 14)

add_executable(gdal_org tutorial_1/gdal_org_read_and_write.cpp)
add_executable(translate tutorial_2/translate.cpp)
add_executable(polygon_op tutorial_3/polygon_op.cpp)
add_executable(gdal_opengl tutorial_4/gdal_opengl.cpp)

target_link_libraries(gdal_org gdal)
target_link_libraries(translate gdal)
target_link_libraries(polygon_op gdal)
target_link_libraries(gdal_opengl gdal GL EGL)