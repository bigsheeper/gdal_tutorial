#include <vector>
#include <cstdio>

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
           const int &height);