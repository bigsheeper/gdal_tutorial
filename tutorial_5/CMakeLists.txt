add_library(geo_handler geo_handler.cu)

include_directories(${CUDA_INCLUDE})
link_directories(${CUDA_LIB})

set_property(TARGET geo_handler PROPERTY CUDA_STANDARD 11)
target_link_libraries(geo_handler
        gdal
        GL
        EGL
        cudart)

add_executable(tutorial_5 main.cpp)
target_link_libraries(tutorial_5 geo_handler)