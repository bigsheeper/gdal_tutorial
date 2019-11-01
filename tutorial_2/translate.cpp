#include <ogrsf_frmts.h>
#include <gdal_utils.h>
#include <iostream>


void Translate() {
    GDALAllRegister();

    auto poDS = GDALOpenEx("point.shp", GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) {
        printf("Open failed.\n");
        exit(1);
    }

    char** papszArgv = nullptr;

    // Spatial query extents, in the SRS of the source layer(s) (or the one specified with -spat_srs).
    // Only features whose geometry intersects the extents will be selected.
    // The geometries will not be clipped unless -clipsrc is specified.
    // argv: -spat <xmin> <ymin> <xmax> <ymax>
    papszArgv = CSLAddString(papszArgv, "-spat");
    papszArgv = CSLAddString(papszArgv, "0");
    papszArgv = CSLAddString(papszArgv, "0");
    papszArgv = CSLAddString(papszArgv, "50");
    papszArgv = CSLAddString(papszArgv, "70");

    auto psOptions = GDALVectorTranslateOptionsNew(papszArgv, nullptr);
    std::string pszDest = "point_translate.shp";
    auto outDS = GDALVectorTranslate(pszDest.c_str(), nullptr, 1, &poDS, psOptions, nullptr);

    if (outDS == NULL) {
        printf("Translate failed.\n");
        exit(1);
    }

    auto poLayer = ((GDALDataset *)outDS)->GetLayerByName("point_translate");

    OGRFeature *poFeature;
    poLayer->ResetReading();
    while ((poFeature = poLayer->GetNextFeature()) != NULL) {
        auto poGeometry = poFeature->GetGeometryRef();
        if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbPoint) {
            auto poPoint = poGeometry->toPoint();
            std::cout << poGeometry->getGeometryName() << "[ " << poPoint->getX() << ", " << poPoint->getY() << " ]" << std::endl;
        } else {
            printf("no point geometry\n");
        }
        OGRFeature::DestroyFeature(poFeature);
    }

    GDALVectorTranslateOptionsFree(psOptions);
    GDALClose(poDS);
    GDALClose(outDS);
}

int main() {
    Translate();
    return 0;
}