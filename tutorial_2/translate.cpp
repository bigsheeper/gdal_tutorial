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

    // Clip input layer with a bounding box.
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
        std::string po_json = poGeometry->exportToJson();
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

void TransQingdao() {
    GDALAllRegister();

    auto poDS = GDALOpenEx("../tutorial_2/qingdao_all.geojson", GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) {
        printf("Open failed.\n");
        exit(1);
    }

    char** papszArgv = nullptr;

    // Clip input layer with a bounding box.
    // argv: -spat <xmin> <ymin> <xmax> <ymax>
    papszArgv = CSLAddString(papszArgv, "-spat");
    papszArgv = CSLAddString(papszArgv, "120.402537");
    papszArgv = CSLAddString(papszArgv, "36.115251");
    papszArgv = CSLAddString(papszArgv, "120.42221");
    papszArgv = CSLAddString(papszArgv, "36.124129");

    auto psOptions = GDALVectorTranslateOptionsNew(papszArgv, nullptr);
    std::string pszDest = "../tutorial_2/qingdao_part.geojson";
    auto outDS = GDALVectorTranslate(pszDest.c_str(), nullptr, 1, &poDS, psOptions, nullptr);

    if (outDS == NULL) {
        printf("Translate failed.\n");
        exit(1);
    }

    GDALVectorTranslateOptionsFree(psOptions);
    GDALClose(poDS);
    GDALClose(outDS);
};

int main() {
//    Translate();
    TransQingdao();
    return 0;
}