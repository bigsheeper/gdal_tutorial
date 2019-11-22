#include <ogrsf_frmts.h>
#include <gdal_utils.h>
#include <iostream>
#include <EGL/egl.h>
#include <GL/gl.h>


#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

struct BoundBox {
    double longitude_left;
    double latitude_left;
    double longitude_right;
    double latitude_right;
};

class GeoHandler {
 public:
    int DiscreteTransX(double longitude);
    int DiscreteTransY(double latitude);

 public:
    int width_;
    int height_;
    BoundBox bound_box_;
};

int GeoHandler::DiscreteTransX(double longitude) {
    double x_left = bound_box_.longitude_left * 111319.490778;
    double x_right = bound_box_.longitude_right * 111319.490778;
    double x_pos = longitude * 111319.490778;
    int ret = (int) (((x_pos - x_left) / (x_right - x_left)) * width_ - 1E-9);
    if (ret < 0) ret = 0;
    if (ret >= width_) ret = width_ - 1;
    return ret;
}


int GeoHandler::DiscreteTransY(double latitude) {
    double y_left = 6378136.99911 * log(tan(.00872664626 * bound_box_.latitude_left + .785398163397));
    double y_right = 6378136.99911 * log(tan(.00872664626 * bound_box_.latitude_right + .785398163397));
    double y_pos = 6378136.99911 * log(tan(.00872664626 * latitude + .785398163397));
    int ret = (int) (((y_pos - y_left) / (y_right - y_left)) * height_ - 1E-9);
    if (ret < 0) ret = 0;
    else if (ret >= height_) ret = height_ - 1;
    return ret;
}

class CityGeo {
 public:
    void Init();

    void Translate();

    void Read();

 public:
    const OGRMultiPolygon &
    buildings() const { return buildings_; }

 public:
    std::string file_path_;
    OGRMultiPolygon buildings_;
    BoundBox bound_box_;
};

void CityGeo::Init() {
    GDALAllRegister();
}

void CityGeo::Translate() {

    auto poDS = GDALOpenEx(file_path_.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) {
        printf("Open failed.\n");
        exit(1);
    }

    char** papszArgv = nullptr;
    std::string shape_type = "ESRI Shapefile";
    // Clip input layer with a bounding box.
    // argv: -spat <xmin> <ymin> <xmax> <ymax>
    papszArgv = CSLAddString(papszArgv, "-spat");
    papszArgv = CSLAddString(papszArgv, std::to_string(bound_box_.longitude_left).c_str());
    papszArgv = CSLAddString(papszArgv, std::to_string(bound_box_.latitude_left).c_str());
    papszArgv = CSLAddString(papszArgv, std::to_string(bound_box_.longitude_right).c_str());
    papszArgv = CSLAddString(papszArgv, std::to_string(bound_box_.latitude_right).c_str());
    papszArgv = CSLAddString(papszArgv, "-f");
    papszArgv = CSLAddString(papszArgv, shape_type.c_str());

    auto psOptions = GDALVectorTranslateOptionsNew(papszArgv, nullptr);
    std::string pszDest = "/tmp/tmp.shp";
    auto outDS = GDALVectorTranslate(pszDest.c_str(), nullptr, 1, &poDS, psOptions, nullptr);

    if (outDS == NULL) {
        printf("Translate failed.\n");
        exit(1);
    }

    GDALVectorTranslateOptionsFree(psOptions);
    GDALClose(poDS);
    GDALClose(outDS);
    CSLDestroy(papszArgv);
}


void CityGeo::Read() {

    auto poDS = GDALOpenEx("/tmp/tmp.shp", GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
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
            buildings_.addGeometry(poPolygon);
        } else if (poGeometry != nullptr && wkbFlatten(poGeometry->getGeometryType()) == wkbMultiPolygon) {
            auto multiPolygon = poGeometry->toMultiPolygon();
            for (auto polygon : multiPolygon) {
                buildings_.addGeometry(polygon);
            }
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
    OGRMultiPolygon buildings_;
    BoundBox bound_box_;
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

    GeoHandler geo_handler;
    geo_handler.width_ = width_;
    geo_handler.height_ = height_;
    geo_handler.bound_box_ = bound_box_;

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ZERO);

    glColor4f(74/255.0, 110/255.0, 121/255.0, 0.5);
    glColor4f(0.0, 0.0, 1.0, 0.65);
    glEnableClientState(GL_VERTEX_ARRAY);

    // draw polygon
    for (auto &building : buildings_) {
        auto ring = building->getExteriorRing();
        auto size = ring->getNumPoints();
        auto points = (OGRRawPoint *)malloc(sizeof(OGRPoint) * size);
        ring->getPoints(points);
        glBegin(GL_POLYGON);
        for (int i = 0; i < size; i++) {
            auto x = geo_handler.DiscreteTransX(points[i].x);
            auto y = geo_handler.DiscreteTransY(points[i].y);
            if(x == 0 || x == width_ - 1 || y == 0 || y == height_ - 1)
                break;
            glVertex2d(x, y);
        }
        glEnd();
    }

    // draw line
    glColor4f(221/255.0, 222/255.0, 217/255.0, 1.0);
    glLineWidth(1);
    glEnable(GL_LINE_SMOOTH);
    for (auto &building : buildings_) {
        auto ring = building->getExteriorRing();
        auto size = ring->getNumPoints();
        auto points = (OGRRawPoint *)malloc(sizeof(OGRPoint) * size);
        ring->getPoints(points);
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i < size; i++) {
            auto x = geo_handler.DiscreteTransX(points[i].x);
            auto y = geo_handler.DiscreteTransY(points[i].y);
            if(x == 0 || x == width_ - 1 || y == 0 || y == height_ - 1)
                break;
            glVertex2d(x, y);
        }
        glEnd();
    }


    glColor4f(1.0, 0.0, 0.0, 1.0);
    glLineWidth(2);
    glEnable(GL_LINE_SMOOTH);
    glBegin(GL_LINE_STRIP);
        glVertex2d(100, 100);
        glVertex2d(500, 100);
        glVertex2d(500, 500);
        glVertex2d(100, 500);
        glVertex2d(100, 100);
    glEnd();
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

int main() {

//    BoundBox bound_box{-74.01695, 40.701673, -73.97243, 40.722044};
//    std::string file_path = "/home/sheep/Downloads/nyc_building/geo_export_6f08c4fb-6554-4408-98c0-1ec36fae8c88.shp";

    BoundBox bound_box{-74.287133,40.480577, -73.678298,40.961747};
    std::string file_path = "/home/sheep/Downloads/nyu_2451_34498/nyu_2451_34498_4326.shp";

//    BoundBox bound_box{-74.287133,40.480577, -73.678298,40.961747};
//    std::string file_path = "/home/sheep/Downloads/nyu_2451_34505/nyu_2451_34505_4326.shp";

    float width = 1328;
    float height = 1110;
    std::string output_path = "offscreen.png";


    CityGeo city_geo;
    city_geo.file_path_ = file_path;
    city_geo.bound_box_ = bound_box;
    city_geo.Init();
    city_geo.Translate();
    city_geo.Read();

    Window2D window_2d;
    window_2d.width_ = width;
    window_2d.height_ = height;
    window_2d.bound_box_ = bound_box;

    window_2d.buildings_ = city_geo.buildings();

    window_2d.Init();
    window_2d.Render();
    window_2d.Finalize();
    window_2d.Terminate();

    window_2d.output_path_ = output_path;
    window_2d.Output();
    return 0;
}

