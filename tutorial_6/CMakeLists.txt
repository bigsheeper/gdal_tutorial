add_library(building_count raw_polygon.cpp handler.cpp point_in_polygon.cu)

include_directories(${CUDA_INCLUDE})
link_directories(${CUDA_LIB})

set_property(TARGET building_count PROPERTY CUDA_STANDARD 11)
target_link_libraries(building_count
        gdal
        GL
        EGL
        cudart)

add_executable(tutorial_6 main.cpp)
target_link_libraries(tutorial_6 building_count)