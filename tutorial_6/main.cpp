#include <iostream>
#include <bitset>
#include <algorithm>
#include "raw_polygon.h"
#include "bounding_box.h"
#include "handler.h"

#define GENERATE_POINTS_NUM 100000

WindowParams window_params{1900, 1410};
const std::string file_path = "/home/sheep/Downloads/nyc_building/nyc_trans.shp";
//const std::string file_path = "/home/sheep/Downloads/nyc_building/geo_export_6f08c4fb-6554-4408-98c0-1ec36fae8c88.shp";
BoundingBox bounding_box{-74.01695, 40.701673, -73.97243, 40.722044};

const std::vector<int>
GenerateSrcX() {
    std::vector<int> x;
    for (int i = 0; i < GENERATE_POINTS_NUM; i++) {
        x.emplace_back(random()%window_params.width);
    }
    std::sort(x.begin(), x.end());
    return x;
}

const std::vector<int>
GenerateSrcY() {
    std::vector<int> y;
    for (int i = 0; i < GENERATE_POINTS_NUM; i++) {
        y.emplace_back(random()%window_params.height);
    }
    return y;
}

const std::vector<int>
GenerateSrcC() {
    std::vector<int> c;
    for (int i = 0; i < GENERATE_POINTS_NUM; i++) {
        c.emplace_back(random()%20);
    }
    return c;
}

int main() {

    RawPolygon raw_polygon;
    raw_polygon.set_file_path(file_path);
    raw_polygon.set_bounding_box(bounding_box);
    raw_polygon.set_window_params(window_params);
    raw_polygon.Extract();
    raw_polygon.TransForm();

    Handler handler;
    handler.set_raw_polygons_xs(raw_polygon.raw_polygons_xs());
    handler.set_raw_polygons_ys(raw_polygon.raw_polygons_ys());
    handler.set_src_vertices_x(GenerateSrcX());
    handler.set_src_vertices_y(GenerateSrcY());
    handler.set_src_vertices_c(GenerateSrcC());
    handler.GetRectangleBoxes();
    handler.Filter();
    handler.Calculate();

    for (int i = 0; i < handler.vertices_x().size(); i++) {
        std::cout << "******************group[ " << i << " ]******************" << std::endl;
        for (int j = 0; j < handler.vertices_x()[i].size(); j++) {
            if (handler.vertices_x()[i][j] != 0) {
                std::cout << handler.vertices_x()[i][j] << "----" << handler.vertices_y()[i][j] << std::endl;
            }
        }
    }

    for (const auto & i : handler.result()) {
        for (unsigned char j : i) {
            if (j != 0)
            std::cout << std::bitset<8>(j) << std::endl;
        }
    }

    return 0;
}

