#include <vector>
#include <cstdint>


class Handler {
 public:
    void
    GetRectangleBoxes();

    void
    Filter();

    void
    Calculate();

 public:
    void
    set_src_vertices_x(const std::vector<int> &src_vertices_x) {
        src_vertices_x_ = src_vertices_x;
    }

    void
    set_src_vertices_y(const std::vector<int> &src_vertices_y) {
        src_vertices_y_ = src_vertices_y;
    }

    void
    set_src_vertices_c(const std::vector<int> &src_vertices_c) {
        src_vertices_c_ = src_vertices_c;
    }

    void
    set_raw_polygons_xs(const std::vector<std::vector<int>> &raw_polygons_xs) {
        raw_polygons_xs_ = raw_polygons_xs;
    }

    void
    set_raw_polygons_ys(const std::vector<std::vector<int>> &raw_polygons_ys) {
        raw_polygons_ys_ = raw_polygons_ys;
    }

    const std::vector<std::vector<int>> &
    vertices_x() const { return vertices_x_; }

    const std::vector<std::vector<int>> &
    vertices_y() const { return vertices_y_; }

    const std::vector<std::vector<int>> &
    vertices_c() const { return vertices_c_; }

    const std::vector<std::vector<uint8_t>> &
    result() const { return result_; }

 private:
    struct RectangleBox  {
        int min_x;
        int min_y;
        int max_x;
        int max_y;
    };

 private:
    std::vector<int> src_vertices_x_;
    std::vector<int> src_vertices_y_;
    std::vector<int> src_vertices_c_;

    std::vector<std::vector<int>> vertices_x_;
    std::vector<std::vector<int>> vertices_y_;
    std::vector<std::vector<int>> vertices_c_;

    std::vector<RectangleBox> rectangle_boxes_;

    std::vector<std::vector<int>> raw_polygons_xs_;
    std::vector<std::vector<int>> raw_polygons_ys_;

    std::vector<std::vector<uint8_t>> result_;
};



