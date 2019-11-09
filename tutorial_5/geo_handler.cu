#include <ogrsf_frmts.h>
#include <gdal_utils.h>
#include <fstream>
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>



/************************************************************************/
/**                                Invoker                             **/
/************************************************************************/
class Invoker {
 public:
    void
    PointsInPolygon();

 public:
    void
    set_point_x(std::vector<double> &point_x) { point_x_ = point_x; }

    void
    set_point_y(std::vector<double> &point_y) { point_y_ = point_y; }

    void
    set_polygon_vertex_x(std::vector<double> &polygon_vertex_x) { polygon_vertex_x_ = polygon_vertex_x; }

    void
    set_polygon_vertex_y(std::vector<double> &polygon_vertex_y) { polygon_vertex_y_ = polygon_vertex_y; }

    const int64_t &
    count() const { return count_; }

 private:
    __global__ void
    PointsInPolygonKernel(const double *dev_point_x,
                          const double *dev_point_y,
                          int64_t num_points,
                          const double *dev_polygon_vertex_x,
                          const double *dev_polygon_vertex_y,
                          int64_t num_polygon_vertexes,
                          int64_t *count);

 private:
    std::vector<double> point_x_;
    std::vector<double> point_y_;

    std::vector<double> polygon_vertex_x_;
    std::vector<double> polygon_vertex_y_;

    int64_t count_;
};


void Invoker::PointsInPolygon() {

    double *dev_point_x;
    double *dev_point_y;
    cudaMalloc(&dev_point_x, sizeof(double) * point_x_.size());
    cudaMalloc(&dev_point_y, sizeof(double) * point_y_.size());
    cudaMemcpy(dev_point_x, &point_x_[0], sizeof(double) * point_x_.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_point_y, &point_y_[0], sizeof(double) * point_y_.size(), cudaMemcpyHostToDevice);

    double *dev_polygon_vertex_x;
    double *dev_polygon_vertex_y;
    cudaMalloc(&dev_polygon_vertex_x, sizeof(double) * polygon_vertex_x_.size());
    cudaMalloc(&dev_polygon_vertex_y, sizeof(double) * polygon_vertex_y_.size());
    cudaMemcpy(dev_polygon_vertex_x,
               &polygon_vertex_x_[0],
               sizeof(double) * polygon_vertex_x_.size(),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(dev_polygon_vertex_y,
               &polygon_vertex_y_[0],
               sizeof(double) * polygon_vertex_y_.size(),
               cudaMemcpyDeviceToHost);

    auto shared_memory_bytes_requirement = sizeof(double) * 2 * polygon_vertex_y_.size();
    int grid_dim = 255;
    int block_dim = 1024;

    PointsInPolygonKernel << < grid_dim, block_dim, shared_memory_bytes_requirement >> > (dev_point_x,
        dev_point_y,
        point_x_.size(),
        dev_polygon_vertex_x,
        dev_polygon_vertex_y,
        polygon_vertex_y_.size(),
        &count_);
}


__host__ __device__ inline double
IsLeft(double x1, double y1, double x2, double y2) {
    return x1 * y2 - x2 * y1;
}


__global__ void
Invoker::PointsInPolygonKernel(const double *dev_point_x,
                               const double *dev_point_y,
                               int64_t num_points,
                               const double *dev_polygon_vertex_x,
                               const double *dev_polygon_vertex_y,
                               int64_t num_polygon_vertexes,
                               int64_t *count) {

    // reference: http://geomalgorithms.com/a03-_inclusion.html
    extern __shared__ char shared_mem[];
    double *poly_xs = (double *) shared_mem;
    double *poly_ys = poly_xs + num_polygon_vertexes;
    for (auto load_index = threadIdx.x; load_index < num_polygon_vertexes;
         load_index += blockDim.x) {
        poly_xs[load_index] = dev_polygon_vertex_x[load_index];
        poly_ys[load_index] = dev_polygon_vertex_y[load_index];
    }
    __syncthreads();
    auto index = threadIdx.x + blockDim.x * blockIdx.x;
    for (; index < num_points; index += blockDim.x * gridDim.x) {

        int winding_num = 0;
        double pnt_x = dev_point_x[index];
        double pnt_y = dev_point_y[index];
        double dx2 = poly_xs[num_polygon_vertexes - 1] - pnt_x;
        double dy2 = poly_ys[num_polygon_vertexes - 1] - pnt_y;
        for (int poly_idx = 0; poly_idx < num_polygon_vertexes; ++poly_idx) {
            auto dx1 = dx2;
            auto dy1 = dy2;
            dx2 = poly_xs[poly_idx] - pnt_x;
            dy2 = poly_ys[poly_idx] - pnt_y;
            bool ref = dy1 < 0;
            if (ref != (dy2 < 0)) {
                if (IsLeft(dx1, dy1, dx2, dy2) < 0 != ref) {
                    winding_num += ref ? 1 : -1;
                }
            }
        }
        if (winding_num != 0) {
            (*count)++;
        }
    }
}

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


void PointsLoader::LoadPoints() {

    std::fstream point_x(pickup_longitude_path_, std::ios_base::in);
    double x;
    while (point_x >> x) {
        points_x_.emplace_back(x);
    }

    std::fstream point_y(pickup_latitude_path_, std::ios_base::in);
    double y;
    while (point_y >> y) {
        points_y_.emplace_back(y);
    }
}


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


void PolygonsLoader::LoadPolygons() {
    GDALAllRegister();

    auto poDS = GDALOpenEx(file_path_.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (poDS == nullptr) {
        printf("Open failed.\n");
        exit(1);
    }

    auto poLayer = ((GDALDataset *) poDS)->GetLayer(0);

    OGRFeature *poFeature;
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
                polygon_vertex_x[2 * i] = points[i].x;
                polygon_vertex_y[2 * i + 1] = points[i].y;
            }
            Polygon polygon{polygon_vertex_x, polygon_vertex_y};
            polygons_.emplace_back(polygon);
        } else {
            printf("no geometry\n");
        }
        OGRFeature::DestroyFeature(poFeature);
    }

    GDALClose(poDS);
}


/************************************************************************/
/**                                Handler                             **/
/************************************************************************/
class Handler {
    void
    Handle();

 private:
    std::vector<PolygonsLoader::Polygon> polygons_;
    std::vector<double> points_x_;
    std::vector<double> points_y_;
    std::vector<int64_t> counts_;
};

void Handler::Handle() {

    counts_.resize(polygons_.size());

    for (int i = 0; i < polygons_.size(); i++) {
        Invoker invoker;
        invoker.set_point_x(points_x_);
        invoker.set_point_y(points_y_);
        invoker.set_polygon_vertex_x(polygons_[i].polygon_vertex_x_);
        invoker.set_polygon_vertex_y(polygons_[i].polygon_vertex_y_);
        invoker.PointsInPolygon();
        counts_[i] = invoker.count();
    }
}


/************************************************************************/
/**                             FeatureWriter                          **/
/************************************************************************/
class FeatureWriter {
 public:
    void
    Write();

 private:
    std::vector<int64_t> counts_;
    std::string file_path_;
};

void FeatureWriter::Write() {

    GDALAllRegister();

    auto poDS = GDALOpenEx(file_path_.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (poDS == nullptr) {
        printf("Open failed.\n");
        exit(1);
    }

    auto poLayer = ((GDALDataset *) poDS)->GetLayer(0);

    OGRFeature *poFeature;
    poLayer->ResetReading();

    int index = 0;
    while ((poFeature = poLayer->GetNextFeature()) != nullptr) {
        auto poGeometry = poFeature->GetGeometryRef();
        if (poGeometry != nullptr && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon) {
            poFeature->SetField("count", (GInt64)counts_[index]);
        } else {
            printf("no geometry\n");
        }
        OGRFeature::DestroyFeature(poFeature);
    }

    GDALClose(poDS);
}


/************************************************************************/
/**                                main()                              **/
/************************************************************************/
int main() {
    // file download: https://data.cityofnewyork.us/Housing-Development/Shapefiles-and-base-map/2k7f-6s2k
    std::string file_path = "/home/sheep/Downloads/nyc_building/geo_export_6f08c4fb-6554-4408-98c0-1ec36fae8c88.shp";
    std::string pickup_longitude = "/home/sheep/Downloads/nyc_building/pickup_longitude.csv";
    std::string pickup_latitude = "/home/sheep/Downloads/nyc_building/pickup_latitude.csv";

    PointsLoader points_loader;
    points_loader.set_file_path(pickup_longitude, pickup_latitude);
    points_loader.LoadPoints();

    return 0;
}