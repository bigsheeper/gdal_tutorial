#include <iostream>
#include "raw_polygon.h"
#include "kernel.h"


void RawPolygon::TransForm() {
    double x_left = bounding_box_.longitude_left * 111319.490778;
    double x_right = bounding_box_.longitude_right * 111319.490778;

    double y_left = 6378136.99911 * log(tan(.00872664626 * bounding_box_.latitude_left + .785398163397));
    double y_right = 6378136.99911 * log(tan(.00872664626 * bounding_box_.latitude_right + .785398163397));

//    CoordTrans(polygons_xs_,
//               polygons_ys_,
//               raw_polygons_xs_,
//               raw_polygons_ys_,
//               x_left,
//               x_right,
//               y_left,
//               y_right,
//               window_params_.width,
//               window_params_.height);

    raw_polygons_xs_.resize(polygons_xs_.size());
    raw_polygons_ys_.resize(polygons_ys_.size());
    for (int i = 0; i < polygons_xs_.size(); i++) {
        raw_polygons_xs_[i].resize(polygons_xs_[i].size());
        raw_polygons_ys_[i].resize(polygons_ys_[i].size());
        for (int j = 0; j < polygons_xs_[i].size(); j++) {
            double x_pos = polygons_xs_[i][j] * 111319.490778;
            int ret_x = (int) (((x_pos - x_left) / (x_right - x_left)) * window_params_.width - 1E-9);
            if (ret_x < 0) ret_x = 0;
            if (ret_x >= window_params_.width) ret_x = window_params_.width - 1;
            raw_polygons_xs_[i][j] = ret_x;

            double y_pos = 6378136.99911 * log(tan(.00872664626 * polygons_ys_[i][j] + .785398163397));
            int ret_y = (int) (((y_pos - y_left) / (y_right - y_left)) * window_params_.height - 1E-9);
            if (ret_y < 0) ret_y = 0;
            else if (ret_y >= window_params_.height) ret_y = window_params_.height - 1;
            raw_polygons_ys_[i][j] = ret_y;
        }
    }

//    raw_polygons_xs_.resize(polygons_xs_.size());
//    raw_polygons_ys_.resize(polygons_ys_.size());
//    for (int i = 0; i < polygons_xs_.size(); i++) {
//        raw_polygons_xs_[i].resize(polygons_xs_[i].size());
//        raw_polygons_ys_[i].resize(polygons_ys_[i].size());
//        for (int j = 0; j < polygons_xs_[i].size(); j++) {
//            raw_polygons_xs_[i][j] = (int)polygons_xs_[i][j];
//            raw_polygons_ys_[i][j] = (int)polygons_ys_[i][j];
//        }
//    }
}

void RawPolygon::Extract() {
    GDALAllRegister();

    auto poDS = GDALOpenEx(file_path_.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (poDS == nullptr) {
        printf("Open failed.\n");
        exit(1);
    }

    auto poLayer = ((GDALDataset *) poDS)->GetLayer(0);

    OGRFeature *poFeature;

    poLayer->ResetReading();
    int64_t building_nums = 0;
    while ((poFeature = poLayer->GetNextFeature()) != nullptr) {
        building_nums++;
    }
    if (building_nums > 10000) {
        std::cout << "too many buildings.";
        return;
    }

    poLayer->ResetReading();
    while ((poFeature = poLayer->GetNextFeature()) != nullptr) {
        auto poGeometry = poFeature->GetGeometryRef();
        if (poGeometry != nullptr && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon) {
            auto poPolygon = poGeometry->toPolygon();
            auto ring = poPolygon->getExteriorRing();
            auto size = ring->getNumPoints();
            auto points = (OGRRawPoint *) malloc(sizeof(OGRRawPoint) * size);
            ring->getPoints(points);
            std::vector<double> polygon_vertex_x(size);
            std::vector<double> polygon_vertex_y(size);
            for (int i = 0; i < size; i++) {
                polygon_vertex_x[i] = points[i].x;
                polygon_vertex_y[i] = points[i].y;
            }
            polygons_xs_.emplace_back(polygon_vertex_x);
            polygons_ys_.emplace_back(polygon_vertex_y);
            free(points);
        } else {
            printf("no geometry\n");
        }
        OGRFeature::DestroyFeature(poFeature);
    }

    GDALClose(poDS);
}

