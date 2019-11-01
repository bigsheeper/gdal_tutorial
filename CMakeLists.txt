cmake_minimum_required(VERSION 3.14)
project(gdal_test)

set(CMAKE_CXX_STANDARD 14)

add_executable(gdal_org_test tutorial_1/gdal_org_read_and_write.cpp)
add_executable(translate tutorial_2/translate.cpp)

target_link_libraries(gdal_org_test gdal)
target_link_libraries(translate gdal)