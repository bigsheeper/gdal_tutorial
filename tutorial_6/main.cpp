#include <iostream>
#include <bitset>
#include <algorithm>
#include <EGL/egl.h>
#include <GL/gl.h>
#include "raw_polygon.h"
#include "bounding_box.h"
#include "handler.h"

#define GENERATE_POINTS_NUM 10000
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

class Window2D {
 public:
    void Init();

    void Render();

    void Terminate();

    void Finalize();

    void Output();

 public:
    float width_;
    float height_;

 public:
    EGLDisplay egl_dpy_;
    EGLSurface egl_surf_;
    EGLContext egl_context_;

 public:
    std::string output_path_;
    unsigned char *buffer_;
    std::vector<std::vector<int>> raw_polygons_xs_;
    std::vector<std::vector<int>> raw_polygons_ys_;

    std::vector<std::vector<int>> vertices_x_;
    std::vector<std::vector<int>> vertices_y_;

    std::vector<Handler::RectangleBox> rectangle_boxes_;
};

void Window2D::Init() {
    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };

    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, (int) width_,
        EGL_HEIGHT, (int) height_,
        EGL_NONE,
    };

    const EGLint contextAttribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE,
    };

    auto &eglDpy = egl_dpy_;
    eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    EGLint major, minor;
    eglInitialize(eglDpy, &major, &minor);

    EGLint numConfigs;
    EGLConfig eglCfg;
    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

    auto &eglSurf = egl_surf_;
    eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);

    eglBindAPI(EGL_OPENGL_API);

    auto &eglCtx = egl_context_;
    eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, contextAttribs);

    eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);
    glOrtho(0, width_, 0, height_, -1, 1);
}

void Window2D::Render() {

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor4f(0.0, 0.0, 1.0, 0.65);
    glEnableClientState(GL_VERTEX_ARRAY);

    // draw buildings
    for (int i = 0; i < raw_polygons_xs_.size(); i++) {
        glBegin(GL_POLYGON);
        for (int j = 0; j < raw_polygons_xs_[i].size(); j++) {
            glVertex2d(raw_polygons_xs_[i][j], raw_polygons_ys_[i][j]);
        }
        glEnd();
    }

    // draw rectangles
//    glColor4f(1.0, 0.0, 0.0, 0.8);
//    glLineWidth(3);
//    for (int i = 0; i < rectangle_boxes_.size(); i++) {
//        glBegin(GL_LINES);
//        glVertex2d(rectangle_boxes_[i].min_x, rectangle_boxes_[i].min_y);
//        glVertex2d(rectangle_boxes_[i].max_x, rectangle_boxes_[i].min_y);
//        glVertex2d(rectangle_boxes_[i].max_x, rectangle_boxes_[i].max_y);
//        glVertex2d(rectangle_boxes_[i].max_x, rectangle_boxes_[i].min_y);
//        glVertex2d(rectangle_boxes_[i].min_x, rectangle_boxes_[i].min_y);
//        glEnd();
//    }

    //draw points
    glPointSize(2);
    glColor4f(0.0, 1.0, 0.0, 0.8);
    for (int i = 0; i < vertices_x_.size(); i++) {
        for (int j = 0; j < vertices_x_[i].size(); j++) {
            glBegin(GL_POINTS);
            glVertex2d(vertices_x_[i][j], vertices_y_[i][j]);
            glEnd();
        }
    }
}

void Window2D::Finalize() {
    eglSwapBuffers(egl_dpy_, egl_surf_);
    buffer_ = (unsigned char *) calloc(4, width_ * height_);
    glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, buffer_);
}

void Window2D::Terminate() {
    eglMakeCurrent(egl_dpy_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(egl_dpy_, egl_context_);
    eglDestroySurface(egl_dpy_, egl_surf_);
    eglReleaseThread();
    eglTerminate(egl_dpy_);
}

void Window2D::Output() {
    auto pixels = buffer_ + (int) (width_ * 4 * (height_ - 1));
    auto stride_bytes = -(width_ * 4);
    stbi_write_png(output_path_.c_str(), width_, height_, 4, pixels, stride_bytes);
}


const std::string file_path = "/home/sheep/Downloads/nyc_building/geo_export_6f08c4fb-6554-4408-98c0-1ec36fae8c88.shp";
BoundingBox bounding_box{-74.532335,40.50033, -72.92947,41.16844};

//const std::string file_path = "/home/sheep/Downloads/nyc_building/jetro.shp";
//BoundingBox bounding_box{-73.906453, 40.648398, -73.895470, 40.654291};

//const std::string file_path = "/home/sheep/Downloads/nyc_building/nyc_trans.shp";
//BoundingBox bounding_box{-74.01695, 40.701673, -73.97243, 40.722044};

WindowParams window_params{1900, 1410};
const std::string output_path = "/home/sheep/polygon/offscreen.png";

const std::vector<int>
GenerateSrcX() {
    std::vector<int> x(GENERATE_POINTS_NUM);
    for (int i = 0; i < GENERATE_POINTS_NUM; i++) {
        x[i] = random()%window_params.width;
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

#include <chrono>
using namespace std::chrono;

int main() {

    RawPolygon raw_polygon;
    raw_polygon.set_file_path(file_path);
    raw_polygon.set_bounding_box(bounding_box);
    raw_polygon.set_window_params(window_params);
    auto start = high_resolution_clock::now();
    raw_polygon.Extract();

    auto stop1 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop1 - start);
    std::cout << "Extract:" << duration.count() / 1000 << " ms" << std::endl;

    raw_polygon.TransForm();

    auto stop2 = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop2 - stop1);
    std::cout << "TransForm:" << duration.count() / 1000 << " ms" << std::endl;

    Handler handler;
    handler.set_raw_polygons_xs(raw_polygon.raw_polygons_xs());
    handler.set_raw_polygons_ys(raw_polygon.raw_polygons_ys());
    handler.set_src_vertices_x(GenerateSrcX());
    handler.set_src_vertices_y(GenerateSrcY());
    handler.set_src_vertices_c(GenerateSrcC());

    handler.GetRectangleBoxes();

    auto stop3 = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop3 - stop2);
    std::cout << "GetRectangleBoxes:" << duration.count() / 1000 << " ms" << std::endl;

    handler.Filter();

    auto stop4 = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop4 - stop3);
    std::cout << "Filter:" << duration.count() / 1000 << " ms" << std::endl;

    handler.Calculate();

    auto stop5 = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop5 - stop4);
    std::cout << "Calculate:" << duration.count() / 1000 << " ms" << std::endl;

    int test = 0;
    for (int i = 0; i < handler.result().size(); i++) {
        for (int j = 0; j < handler.result()[i].size(); j++) {
            test++;
        }
    }
    std::cout << "test = " << test << std::endl;

    Window2D window_2d;
    window_2d.width_ = window_params.width;
    window_2d.height_ = window_params.height;
    window_2d.raw_polygons_xs_ = raw_polygon.raw_polygons_xs();
    window_2d.raw_polygons_ys_ = raw_polygon.raw_polygons_ys();

    window_2d.vertices_x_ = handler.vertices_x();
    window_2d.vertices_y_ = handler.vertices_y();
    window_2d.rectangle_boxes_ = handler.rectangle_boxes();

    window_2d.Init();
    window_2d.Render();
    window_2d.Finalize();
    window_2d.Terminate();

    window_2d.output_path_ = output_path;
    window_2d.Output();

    return 0;
}

