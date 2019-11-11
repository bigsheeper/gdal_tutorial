#include <iostream>
#include <fstream>
#include <cstdint>

#include <ogrsf_frmts.h>
#include <gdal_utils.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>


/************************************************************************/
/**                                Invoker                             **/
/************************************************************************/
class Invoker {
 public:
    void
    PointsInPolygon();

 public:
    void
    set_point_x(const std::vector<double> &point_x) { point_x_ = point_x; }

    void
    set_point_y(const std::vector<double> &point_y) { point_y_ = point_y; }

    void
    set_polygon_vertex_x(const std::vector<double> &polygon_vertex_x) { polygon_vertex_x_ = polygon_vertex_x; }

    void
    set_polygon_vertex_y(const std::vector<double> &polygon_vertex_y) { polygon_vertex_y_ = polygon_vertex_y; }

    const int64_t &
    count() const { return count_; }

 private:
    std::vector<double> point_x_;
    std::vector<double> point_y_;

    std::vector<double> polygon_vertex_x_;
    std::vector<double> polygon_vertex_y_;

    int64_t count_;
};


/************************************************************************/
/**                             PointsLoader                           **/
/************************************************************************/
class PointsLoader {
 public:
    void
    LoadPoints();

    void
    set_file_path(const std::string &pickup_longitude_path, const std::string &pickup_latitude_path) {
        pickup_longitude_path_ = pickup_longitude_path;
        pickup_latitude_path_ = pickup_latitude_path;
    }

    const std::vector<double> &
    points_x() const { return points_x_; }

    const std::vector<double> &
    points_y() const { return points_y_; }

 private:
    std::string pickup_longitude_path_;
    std::string pickup_latitude_path_;
    std::vector<double> points_x_;
    std::vector<double> points_y_;
};


/************************************************************************/
/**                           PolygonsLoader                           **/
/************************************************************************/
class PolygonsLoader {
 public:
    struct Polygon {
        std::vector<double> polygon_vertex_x_;
        std::vector<double> polygon_vertex_y_;
    };

 public:
    void
    LoadPolygons();

    void
    set_file_path(const std::string &file_path) {
        file_path_ = file_path;
    }

    const std::vector<Polygon> &
    polygons() const { return polygons_; }

 private:
    std::vector<Polygon> polygons_;
    std::string file_path_;
};


/************************************************************************/
/**                                Handler                             **/
/************************************************************************/
class Handler {
 public:
    void
    Handle();

 public:
    void
    set_polygons(const std::vector<PolygonsLoader::Polygon> &polygons) { polygons_ = polygons; }

    void
    set_points_x(const std::vector<double> &points_x) { points_x_ = points_x; }

    void
    set_points_y(const std::vector<double> &points_y) { points_y_ = points_y; }

    const std::vector<int64_t> &
    counts() const { return counts_; }

 private:
    std::vector<PolygonsLoader::Polygon> polygons_;
    std::vector<double> points_x_;
    std::vector<double> points_y_;
    std::vector<int64_t> counts_;
};


/************************************************************************/
/**                             FeatureWriter                          **/
/************************************************************************/
class FeatureWriter {
 public:
    void
    Write();

 public:
    void
    set_counts(const std::vector<int64_t> &counts) { counts_ = counts; }

    void
    set_file_path(const std::string &file_path) { file_path_ = file_path; }

    const std::vector<int64_t> &
    counts() const { return counts_; }

 private:
    std::vector<int64_t> counts_;
    std::string file_path_;
};