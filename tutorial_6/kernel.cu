#include "kernel.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>


__global__ void
CoordTransKernel(const double *dev_x,
                 const double *dev_y,
                 int *dev_out_x,
                 int *dev_out_y,
                 const double x_left,
                 const double x_right,
                 const double y_left,
                 const double y_right,
                 const int width,
                 const int height,
                 const int64_t num_points) {

    auto index = threadIdx.x + blockDim.x * blockIdx.x;
    for (; index < num_points; index += blockDim.x * gridDim.x) {
        double x_pos = dev_x[index] * 111319.490778;
        int ret_x = (int) (((x_pos - x_left) / (x_right - x_left)) * width - 1E-9);
        if (ret_x < 0) ret_x = 0;
        if (ret_x >= width) ret_x = width - 1;
        dev_out_x[index] = ret_x;

        double y_pos = 6378136.99911 * log(tan(.00872664626 * dev_y[index] + .785398163397));
        int ret_y = (int) (((y_pos - y_left) / (y_right - y_left)) * height - 1E-9);
        if (ret_y < 0) ret_y = 0;
        else if (ret_y >= height) ret_y = height - 1;
        dev_out_y[index] = ret_y;
    }
}


void
CoordTrans(const std::vector<std::vector<double>> &polygons_xs,
           const std::vector<std::vector<double>> &polygons_ys,
           std::vector<std::vector<int>> &raw_polygons_xs,
           std::vector<std::vector<int>> &raw_polygons_ys,
           const double &x_left,
           const double &x_right,
           const double &y_left,
           const double &y_right,
           const int &width,
           const int &height) {

    int64_t num_points = 0;
    for (const auto &polygons_x : polygons_xs) {
        num_points += polygons_x.size();
    }

    double *dev_x;
    double *dev_y;
    cudaMalloc(&dev_x, num_points * sizeof(double));
    cudaMalloc(&dev_y, num_points * sizeof(double));

    int offset = 0;
    for (int i = 0; i < polygons_xs.size(); i++) {
        cudaMemcpy(dev_x + offset,
                   &polygons_xs[i][0],
                   polygons_xs[i].size() * sizeof(double),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(dev_y + offset,
                   &polygons_ys[i][0],
                   polygons_ys[i].size() * sizeof(double),
                   cudaMemcpyHostToDevice);

        offset += polygons_xs[i].size() * sizeof(double);
    }

    int *dev_out_x;
    int *dev_out_y;
    cudaMalloc(&dev_out_x, num_points * sizeof(int));
    cudaMalloc(&dev_out_y, num_points * sizeof(int));

    CoordTransKernel << < 256, 1024 >> > (dev_x,
        dev_y,
        dev_out_x,
        dev_out_y,
        x_left,
        x_right,
        y_left,
        y_right,
        width,
        height,
        num_points);
    cudaDeviceSynchronize();

    offset = 0;
    raw_polygons_xs.resize(polygons_xs.size());
    raw_polygons_ys.resize(polygons_ys.size());
    for (int i = 0; i < polygons_xs.size(); i++) {
        raw_polygons_xs[i].resize(polygons_xs[i].size());
        raw_polygons_ys[i].resize(polygons_ys[i].size());
        cudaMemcpy(&raw_polygons_xs[i][0],
                   dev_out_x + offset,
                   polygons_xs[i].size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&raw_polygons_ys[i][0],
                   dev_out_y + offset,
                   polygons_ys[i].size() * sizeof(int),
                   cudaMemcpyDeviceToHost);
        offset += polygons_xs[i].size() * sizeof(int);
    }
}







