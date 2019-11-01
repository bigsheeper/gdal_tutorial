#include <ogrsf_frmts.h>
#include <gdal_utils.h>
#include <iostream>


void ReadFromGeoJson() {
    GDALAllRegister();

    auto poDS = GDALOpenEx("../tutorial_3/qingdao.geojson", GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) {
        printf("Open failed.\n");
        exit(1);
    }

    auto poLayer = ((GDALDataset *) poDS)->GetLayerByName("qingdao");

    OGRFeature *poFeature;
    poLayer->ResetReading();
    int index = 0;
    while ((poFeature = poLayer->GetNextFeature()) != NULL) {
        auto poGeometry = poFeature->GetGeometryRef();
        if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon) {
            auto poPolygon = poGeometry->toPolygon();
            std::cout << "**********index " << index << "**********\n"
                      << poGeometry->getGeometryName() << ": area = "
                      << poPolygon->get_Area() << std::endl;

            auto er = poPolygon->getExteriorRing();
            if (er != nullptr) {
                std::cout << er->getGeometryName() << ":\narea = "
                          << er->get_Area() << ",\nis_closed = "
                          << er->get_IsClosed() << ",\nlength = "
                          << er->get_Length() << ",\nNumPoints = "
                          << er->getNumPoints() << std::endl;
                for (int i = 0; i < er->getNumPoints(); i++) {
                    std::cout << "x = " << er->getY(i) << ", y = " << er->getY(i) << std::endl;
                }
            }

            auto ir = poPolygon->getInteriorRing(index);
            if (ir != nullptr) {
                std::cout << ir->getGeometryName() << ":\narea = "
                          << ir->get_Area() << ",\nis_closed = "
                          << ir->get_IsClosed() << ",\nlength = "
                          << ir->get_Length() << ",\nNumPoints = "
                          << ir->getNumPoints() << std::endl;
            }
        } else {
            printf("no geometry\n");
        }
        OGRFeature::DestroyFeature(poFeature);
        index++;
    }

    GDALClose(poDS);
}

int main() {
    ReadFromGeoJson();
    return 0;
}
