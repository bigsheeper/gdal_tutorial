#include <ogrsf_frmts.h>
#include <gdal_utils.h>
#include <iostream>
#include <EGL/egl.h>
#include <GL/gl.h>


#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"


class GeoHandler {
 public:
    int DiscreteTransX(double longitude);
    int DiscreteTransY(double latitude);

 public:
    int width_;
    int height_;
    double longitude_left_;
    double longitude_right_;
    double latitude_left_;
    double latitude_right_;
};

int GeoHandler::DiscreteTransX(double longitude) {
    double x_left = longitude_left_ * 111319.490778;
    double x_right = longitude_right_ * 111319.490778;
    double x_pos = longitude * 111319.490778;
    int ret = (int) (((x_pos - x_left) / (x_right - x_left)) * width_ - 1E-9);
    if (ret < 0) ret = 0;
    if (ret >= width_) ret = width_ - 1;
    return ret;
}


int GeoHandler::DiscreteTransY(double latitude) {
    double y_left = 6378136.99911 * log(tan(.00872664626 * latitude_left_ + .785398163397));
    double y_right = 6378136.99911 * log(tan(.00872664626 * latitude_right_ + .785398163397));
    double y_pos = 6378136.99911 * log(tan(.00872664626 * latitude + .785398163397));
    int ret = (int) (((y_pos - y_left) / (y_right - y_left)) * height_ - 1E-9);
    if (ret < 0) ret = 0;
    else if (ret >= height_) ret = height_ - 1;
    return ret;
}


class QingDao {
 public:
    void Read();

 public:
    const OGRMultiPolygon &
    buildings() const { return buildings_; }

 public:
    std::string file_path_;
    OGRMultiPolygon buildings_;
};


void QingDao::Read() {
    GDALAllRegister();

    auto poDS = GDALOpenEx(file_path_.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (poDS == nullptr) {
        printf("Open failed.\n");
        exit(1);
    }

    auto poLayer = ((GDALDataset *) poDS)->GetLayerByName("qingdao");

    OGRFeature *poFeature;
    poLayer->ResetReading();
    while ((poFeature = poLayer->GetNextFeature()) != nullptr) {
        auto poGeometry = poFeature->GetGeometryRef();
        if (poGeometry != nullptr && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon) {
            auto poPolygon = poGeometry->toPolygon();
            buildings_.addGeometry(poPolygon);
        } else {
            printf("no geometry\n");
        }
        OGRFeature::DestroyFeature(poFeature);
    }
    GDALClose(poDS);
}


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
    QingDao qing_dao;
    qing_dao.file_path_ = "../tutorial_2/qingdao_part.geojson";
    qing_dao.Read();
    auto &buildings = qing_dao.buildings();

    GeoHandler geo_handler;
    geo_handler.width_ = width_;
    geo_handler.height_ = height_;
    geo_handler.longitude_left_ = 120.402537;
    geo_handler.longitude_right_ = 120.42221;
    geo_handler.latitude_left_ = 36.115251;
    geo_handler.latitude_right_ = 36.124129;

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glColor4f(0.0, 0.0, 1.0, 1.0);
    glEnableClientState(GL_VERTEX_ARRAY);

    for (auto &building : buildings) {
        auto ring = building->getExteriorRing();
        auto size = ring->getNumPoints();
        auto points = (OGRRawPoint *)malloc(sizeof(OGRPoint) * size);
        ring->getPoints(points);
        for (int i = 0; i < size; i++) {
            points[i].x = geo_handler.DiscreteTransX(points[i].x);
            points[i].y = geo_handler.DiscreteTransY(points[i].y);
        }
        glVertexPointer(2, GL_DOUBLE, 0, (double *) points);
        glDrawArrays(GL_POLYGON, 0, size);
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
    stbi_write_png(output_path_.c_str(), width_, height_, 4, buffer_, width_ * 4);
}

int main() {

    Window2D window_2d;
    window_2d.width_ = 1900;
    window_2d.height_ = 1410;

    window_2d.Init();
    window_2d.Render();
    window_2d.Finalize();
    window_2d.Terminate();

    window_2d.output_path_ = "offscreen.png";
    window_2d.Output();
    return 0;
}

