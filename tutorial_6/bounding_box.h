#pragma once

struct BoundingBox {
    double longitude_left;
    double latitude_left;
    double longitude_right;
    double latitude_right;
};

struct WindowParams {
    int width;
    int height;
};