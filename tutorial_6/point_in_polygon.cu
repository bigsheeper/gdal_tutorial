#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cstdio>


__host__ __device__ inline int
isLeft(int x1, int y1, int x2, int y2) {
    return x1 * y2 - x2 * y1;
}

__global__ void
PIP_WindingNumber_kernel(const int* dev_point_xs,
                         const int* dev_point_ys,
                         int64_t num_points,
                         const int* dev_vertex_x,
                         const int* dev_vertex_y,
                         int64_t num_poly_vertexes,
                         uint8_t* __restrict__ output) {

    extern __shared__ char shared_mem[];
    int* poly_xs = (int*)shared_mem;
    int* poly_ys = poly_xs + num_poly_vertexes;
    for (auto load_index = threadIdx.x; load_index < num_poly_vertexes;
         load_index += blockDim.x) {
        poly_xs[load_index] = dev_vertex_x[load_index];
        poly_ys[load_index] = dev_vertex_y[load_index];
    }
    __syncthreads();
    auto index = threadIdx.x + blockDim.x * blockIdx.x;
    for (; index < num_points; index += blockDim.x * gridDim.x) {

        int winding_num = 0;
        int pnt_x = dev_point_xs[index];
        int pnt_y = dev_point_ys[index];
        int dx2 = poly_xs[num_poly_vertexes - 1] - pnt_x;
        int dy2 = poly_ys[num_poly_vertexes - 1] - pnt_y;
        for (int poly_idx = 0; poly_idx < num_poly_vertexes; ++poly_idx) {
            auto dx1 = dx2;
            auto dy1 = dy2;
            dx2 = poly_xs[poly_idx] - pnt_x;
            dy2 = poly_ys[poly_idx] - pnt_y;
            bool ref = dy1 < 0;
            if (ref != (dy2 < 0)) {
                if (isLeft(dx1, dy1, dx2, dy2) < 0 != ref) {
                    winding_num += ref ? 1 : -1;
                }
            }
        }
        uint8_t ans = winding_num != 0;
        output[index] = ans;
    }
}


void
PointInPolygon_naive_fast(const int* point_x,
                          const int* point_y,
                          int64_t num_points,
                          const int* vertex_x,
                          const int* vertex_y,
                          int64_t num_poly_vertexes,
                          uint8_t* __restrict__ output) {

    auto shared_memory_bytes_requirement = sizeof(int) * 2 * num_poly_vertexes;

    PIP_WindingNumber_kernel<<<256, 1024, shared_memory_bytes_requirement>>>(point_x,
                                         point_y,
                                         num_points,
                                         vertex_x,
                                         vertex_y,
                                         num_poly_vertexes,
                                         output);
}


void
PointInPolygon(const int* point_x,
               const int* point_y,
               int64_t num_points,
               const int* vertex_x,
               const int* vertex_y,
               int64_t num_ploygon_vertexes,
               uint8_t* output) {

    PointInPolygon_naive_fast(point_x,
                              point_y,
                              num_points,
                              vertex_x,
                              vertex_y,
                              num_ploygon_vertexes,
                              output);
}
