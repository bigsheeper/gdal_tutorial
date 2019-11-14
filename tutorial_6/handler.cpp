#include <cuda_runtime.h>
#include "algorithm"
#include "handler.h"
#include "point_in_polygon.h"

void Handler::GetRectangleBoxes() {

    rectangle_boxes_.resize(raw_polygons_xs_.size());

    for (int i = 0; i < raw_polygons_xs_.size(); i++) {
        auto min_x = std::min_element(raw_polygons_xs_[i].begin(), raw_polygons_xs_[i].end());
        auto min_y = std::min_element(raw_polygons_ys_[i].begin(), raw_polygons_ys_[i].end());
        auto max_x = std::max_element(raw_polygons_xs_[i].begin(), raw_polygons_xs_[i].end());
        auto max_y = std::max_element(raw_polygons_ys_[i].begin(), raw_polygons_ys_[i].end());
        RectangleBox rectangle_box{*min_x, *min_y, *max_x, *max_y};
        rectangle_boxes_[i] = rectangle_box;
    }
}

void Handler::Filter() {

    vertices_x_.resize(rectangle_boxes_.size());
    vertices_y_.resize(rectangle_boxes_.size());
    vertices_c_.resize(rectangle_boxes_.size());

    int box_index = 0;
    for (auto rectangle_box : rectangle_boxes_) {
        std::vector<int> x;
        std::vector<int> y;
        std::vector<int> c;

        for (int i = 0; i < src_vertices_x_.size(); i++) {
            if (src_vertices_x_[i] < rectangle_box.min_x) {
                continue;
            }
            if (src_vertices_x_[i] > rectangle_box.max_x) {
                break;
            }
            if( src_vertices_y_[i] >= rectangle_box.min_y && src_vertices_y_[i] <= rectangle_box.max_y) {
                x.emplace_back(src_vertices_x_[i]);
                y.emplace_back(src_vertices_y_[i]);
                c.emplace_back(src_vertices_c_[i]);
            }
        }

        vertices_x_[box_index] = x;
        vertices_y_[box_index] = y;
        vertices_c_[box_index++] = c;
    }
}

void Handler::Calculate() {

    result_.resize(raw_polygons_xs_.size());

    for (int i = 0; i < raw_polygons_xs_.size(); i++) {
        int* point_x;
        int* point_y;
        cudaMalloc(&point_x, vertices_x_[i].size() * sizeof(int));
        cudaMalloc(&point_y, vertices_y_[i].size() * sizeof(int));
        cudaMemcpy(point_x, &vertices_x_[i][0], vertices_x_[i].size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(point_y, &vertices_y_[i][0], vertices_y_[i].size() * sizeof(int), cudaMemcpyHostToDevice);

        int* vertex_x;
        int* vertex_y;
        cudaMalloc(&vertex_x, raw_polygons_xs_[i].size() * sizeof(int));
        cudaMalloc(&vertex_y, raw_polygons_ys_[i].size() * sizeof(int));
        cudaMemcpy(vertex_x, &raw_polygons_xs_[i][0], raw_polygons_xs_[i].size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(vertex_y, &raw_polygons_ys_[i][0], raw_polygons_ys_[i].size() * sizeof(int), cudaMemcpyHostToDevice);

        uint8_t* output;
        cudaMalloc(&output, vertices_x_[i].size() * sizeof(uint8_t));

        auto num_points = vertices_x_[i].size();
        auto num_ploygon_vertexes = raw_polygons_xs_[i].size();
        PointInPolygon(point_x, point_y, num_points, vertex_x, vertex_y, num_ploygon_vertexes, output);
        result_[i].resize(vertices_x_[i].size());
        cudaMemcpy(&result_[i][0], output, vertices_x_[i].size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        cudaFree(point_x);
        cudaFree(point_y);
        cudaFree(vertex_x);
        cudaFree(vertex_y);
        cudaFree(output);
    }
}
