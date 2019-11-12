#include "geo_handler.h"

/************************************************************************/
/**                                Invoker                             **/
/************************************************************************/

__host__ __device__ inline double
IsLeft(double x1, double y1, double x2, double y2) {
    return x1 * y2 - x2 * y1;
}


__global__ void
PointsInPolygonKernel(const double *dev_point_x,
                               const double *dev_point_y,
                               int64_t num_points,
                               const double *dev_polygon_vertex_x,
                               const double *dev_polygon_vertex_y,
                               int64_t num_polygon_vertexes,
                               int64_t &count) {

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
//        printf("pnt_x = %lf\n", pnt_x);
        double pnt_y = dev_point_y[index];
//        printf("pnt_y = %lf\n", pnt_y);
        double dx2 = poly_xs[num_polygon_vertexes - 1] - pnt_x;
//        printf("dx2 = %lf\n", dx2);
        double dy2 = poly_ys[num_polygon_vertexes - 1] - pnt_y;
//        printf("dy2 = %lf\n", dy2);
        for (int poly_idx = 0; poly_idx < num_polygon_vertexes; ++poly_idx) {
            auto dx1 = dx2;
//            printf("dx1 = %lf\n", dx1);
            auto dy1 = dy2;
//            printf("dy1 = %lf\n", dy1);
            dx2 = poly_xs[poly_idx] - pnt_x;
//            printf("dx2 = %lf\n", dx2);
            dy2 = poly_ys[poly_idx] - pnt_y;
//            printf("dy2 = %lf\n", dy2);
            bool ref = dy1 < 0;
//            printf("ref = %d\n", ref);
//            printf("dy2 < 0 = %d\n", dy2 < 0);
            if (ref != (dy2 < 0)) {
                if (IsLeft(dx1, dy1, dx2, dy2) < 0 != ref) {
//                    printf("OK\n");
                    winding_num += ref ? 1 : -1;
                }
            }
        }
        if (winding_num != 0) {
//            printf("OK\n");
            count++;
//            printf("count=%ld",count);
        }
    }
//    printf("count=%ld",count);
}


void Invoker::PointsInPolygon() {

    count_ = 0;

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
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_polygon_vertex_y,
               &polygon_vertex_y_[0],
               sizeof(double) * polygon_vertex_y_.size(),
               cudaMemcpyHostToDevice);

    auto shared_memory_bytes_requirement = sizeof(double) * 2 * polygon_vertex_y_.size();
    int grid_dim = 255;
    int block_dim = 1024;

    PointsInPolygonKernel << < grid_dim, block_dim, shared_memory_bytes_requirement >> > (dev_point_x,
        dev_point_y,
        point_x_.size(),
        dev_polygon_vertex_x,
        dev_polygon_vertex_y,
        polygon_vertex_y_.size(),
        count_);

    cudaDeviceSynchronize();
}

/************************************************************************/
/**                             PointsLoader                           **/
/************************************************************************/
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
                polygon_vertex_x[i] = points[i].x;
                polygon_vertex_y[i] = points[i].y;
            }
            Polygon polygon{polygon_vertex_x, polygon_vertex_y};
            polygons_.emplace_back(polygon);
            free(points);
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
            poFeature->SetField("count", (GInt64) counts_[index]);
        } else {
            printf("no geometry\n");
        }
        OGRFeature::DestroyFeature(poFeature);
    }

    GDALClose(poDS);
}


