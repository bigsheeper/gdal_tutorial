#include "geo_handler.h"
#include <chrono>

/************************************************************************/
/**                                main()                              **/
/************************************************************************/
int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // file download: https://data.cityofnewyork.us/Housing-Development/Shapefiles-and-base-map/2k7f-6s2k
    std::string file_path = "/home/sheep/Downloads/nyc_building/geo_export_6f08c4fb-6554-4408-98c0-1ec36fae8c88.shp";
//    std::string file_path = "/home/sheep/Downloads/nyc_building/nyc_trans.shp";
    std::string pickup_longitude = "/home/sheep/Downloads/nyc_building/pickup_longitude.csv";
    std::string pickup_latitude = "/home/sheep/Downloads/nyc_building/pickup_latitude.csv";

    PointsLoader points_loader;
    points_loader.set_file_path(pickup_longitude, pickup_latitude);
    points_loader.LoadPoints();

    PolygonsLoader polygons_loader;
    polygons_loader.set_file_path(file_path);
    polygons_loader.LoadPolygons();

    handler handler;
    handler.set_polygons(polygons_loader.polygons());
    handler.set_points_x(points_loader.points_x());
    handler.set_points_y(points_loader.points_y());
    handler.Handle();

    FeatureWriter feature_writer;
    feature_writer.set_file_path(file_path);
    feature_writer.set_counts(handler.counts());
    std::ofstream myfile;
    myfile.open ("example.txt");
    myfile << "Writing this to a file.\n";
    for (auto count : feature_writer.counts()) {
        myfile << count << "\n";
    }
    myfile.close();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count()/1000 << "ms" << std::endl;
//    feature_writer.Write();

    return 0;
}

