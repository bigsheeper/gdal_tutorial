#pragma once

#include <ogrsf_frmts.h>
#include <gdal_utils.h>

#include "bounding_box.h"

class RawPolygon {
 public:
    void
    Extract();

    void
    TransForm();

 public:
    void
    set_file_path(const std::string &file_path) {
        file_path_ = file_path;
    }

    void
    set_bounding_box(const BoundingBox &bounding_box) {
        bounding_box_ = bounding_box;
    }

    void
    set_window_params(const WindowParams &window_params) {
        window_params_ = window_params;
    }

    const std::vector<std::vector<int>> &
    raw_polygons_xs() const { return raw_polygons_xs_; }

    const std::vector<std::vector<int>> &
    raw_polygons_ys() const { return raw_polygons_ys_; }

    const std::vector<std::vector<double>> &
    polygons_xs() const { return polygons_xs_; }

    const std::vector<std::vector<double>> &
    polygons_ys() const { return polygons_ys_; }

 private:
    std::string file_path_;
    BoundingBox bounding_box_;
    WindowParams window_params_;

    std::vector<std::vector<double>> polygons_xs_;
    std::vector<std::vector<double>> polygons_ys_;

    std::vector<std::vector<int>> raw_polygons_xs_;
    std::vector<std::vector<int>> raw_polygons_ys_;
};