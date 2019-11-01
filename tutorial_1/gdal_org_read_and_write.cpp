#include <iostream>
#include <ogrsf_frmts.h>


void ReadingFromOGR() {
    // 1.Initially it is necessary to register all the format drivers that are desired.
    // This is normally accomplished by calling GDALAllRegister() which registers all format drivers built into GDAL/OGR.
    GDALAllRegister();

    // 2.Next we need to open the input OGR datasource.
    // Datasources can be files, RDBMSes, directories full of files, or even remote web services depending on the driver being used.
    // However, the datasource name is always a single string. In this case we are hardcoded to open a particular shapefile.
    // The second argument (GDAL_OF_VECTOR) tells the OGROpen() method that we want a vector driver to be use and that don’t require update access.
    // On failure NULL is returned, and we report an error.
    GDALDataset *poDS;

    poDS = (GDALDataset *) GDALOpenEx("point.shp", GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) {
        printf("Open failed.\n");
        exit(1);
    }

    // 3.A GDALDataset can potentially have many layers associated with it.
    // The number of layers available can be queried with GDALDataset::GetLayerCount() and individual layers fetched by index using GDALDataset::GetLayer().
    // However, we will just fetch the layer by name.
    OGRLayer *poLayer;

    poLayer = poDS->GetLayerByName("point");

    // 4.Now we want to start reading features from the layer.
    // Before we start we could assign an attribute or spatial filter to the layer to restrict the set of feature we get back, but for now we are interested in getting all features.
    // If using older GDAL versions, while it isn’t strictly necessary in this circumstance since we are starting fresh with the layer,
    // it is often wise to call OGRLayer::ResetReading() to ensure we are starting at the beginning of the layer.
    // We iterate through all the features in the layer using OGRLayer::GetNextFeature(). It will return NULL when we run out of features.
    OGRFeature *poFeature;

    poLayer->ResetReading();
    while ((poFeature = poLayer->GetNextFeature()) != NULL) {
        // In order to dump all the attribute fields of the feature, it is helpful to get the OGRFeatureDefn.
        // This is an object, associated with the layer, containing the definitions of all the fields.
        // We loop over all the fields, and fetch and report the attributes based on their type.
        OGRFeatureDefn *poFDefn = poLayer->GetLayerDefn();
        for (int iField = 0; iField < poFDefn->GetFieldCount(); iField++) {

            OGRFieldDefn *poFieldDefn = poFDefn->GetFieldDefn(iField);
            switch (poFieldDefn->GetType()) {
                case OFTInteger:printf("%d,", poFeature->GetFieldAsInteger(iField));
                    break;
                case OFTInteger64:printf(CPL_FRMT_GIB ",", poFeature->GetFieldAsInteger64(iField));
                    break;
                case OFTReal:printf("%.3f,", poFeature->GetFieldAsDouble(iField));
                    break;
                case OFTString:printf("%s,", poFeature->GetFieldAsString(iField));
                    break;
                default:printf("%s,", poFeature->GetFieldAsString(iField));
                    break;
            }
        }

        // There are a few more field types than those explicitly handled above, but a reasonable representation of them can be fetched with the OGRFeature::GetFieldAsString() method.
        // In fact we could shorten the above by using GetFieldAsString() for all the types.
        //
        // Next we want to extract the geometry from the feature, and write out the point geometry x and y.
        // Geometries are returned as a generic OGRGeometry pointer.
        // We then determine the specific geometry type, and if it is a point, we cast it to point and operate on it.
        // If it is something else we write placeholders.
        OGRGeometry *poGeometry;

        poGeometry = poFeature->GetGeometryRef();
        if (poGeometry != NULL
            && wkbFlatten(poGeometry->getGeometryType()) == wkbPoint) {
            OGRPoint *poPoint = poGeometry->toPoint();
            printf("%.3f,%3.f\n", poPoint->getX(), poPoint->getY());
        } else {
            printf("no point geometry\n");
        }

        // For GDAL > 2.3, as the OGRLayer::GetNextFeature() method returns a copy of the feature that is now owned by us.
        // So at the end of use we must free the feature.
        // We could just “delete” it, but this can cause problems in windows builds where the GDAL DLL has a different “heap” from the main program.
        // To be on the safe side we use a GDAL function to delete the feature.
        OGRFeature::DestroyFeature(poFeature);
    }

    // 5.The OGRLayer returned by GDALDataset::GetLayerByName() is also a reference to an internal layer owned by the GDALDataset so we don’t need to delete it.
    // But we do need to delete the datasource in order to close the input file.
    // Once again we do this with a custom delete method to avoid special win32 heap issues.
    GDALClose(poDS);
}

void WritingToOGR() {

    // 1.We start by registering all the drivers, and then fetch the Shapefile driver as we will need it to create our output file.
    const char *pszDriverName = "ESRI Shapefile";
    GDALDriver *poDriver;

    GDALAllRegister();

    poDriver = GetGDALDriverManager()->GetDriverByName(pszDriverName);
    if (poDriver == NULL) {
        printf("%s driver not available.\n", pszDriverName);
        exit(1);
    }

    // 2.Next we create the datasource.
    // The ESRI Shapefile driver allows us to create a directory full of shapefiles, or a single shapefile as a datasource.
    // In this case we will explicitly create a single file by including the extension in the name.
    // Other drivers behave differently. The second, third, fourth and fifth argument are related to raster dimensions (in case the driver has raster capabilities).
    // The last argument to the call is a list of option values, but we will just be using defaults in this case. Details of the options supported are also format specific.
    GDALDataset *poDS;

    poDS = poDriver->Create("point.shp", 0, 0, 0, GDT_Unknown, NULL);
    if (poDS == NULL) {
        printf("Creation of output file failed.\n");
        exit(1);
    }

    // 3.Now we create the output layer.
    // In this case since the datasource is a single file, we can only have one layer.
    // We pass wkbPoint to specify the type of geometry supported by this layer.
    // In this case we aren’t passing any coordinate system information or other special layer creation options.
    OGRLayer *poLayer;

    poLayer = poDS->CreateLayer("point", NULL, wkbPoint, NULL);
    if (poLayer == NULL) {
        printf("Layer creation failed.\n");
        exit(1);
    }

    // 4.Now that the layer exists, we need to create any attribute fields that should appear on the layer.
    // Fields must be added to the layer before any features are written.
    // To create a field we initialize an OGRField object with the information about the field.
    // In the case of Shapefiles, the field width and precision is significant in the creation of the output .dbf file,
    // so we set it specifically, though generally the defaults are OK.
    // For this example we will just have one attribute, a name string associated with the x,y point.
    //
    // Note that the template OGRField we pass to OGRLayer::CreateField() is copied internally. We retain ownership of the object.
    OGRFieldDefn oField("Name", OFTString);

    oField.SetWidth(32);

    if (poLayer->CreateField(&oField) != OGRERR_NONE) {
        printf("Creating Name field failed.\n");
        exit(1);
    }

    // The following snipping loops reading lines of the form “x,y,name” from stdin, and parsing them.
    double x, y;
    std::string szName;

    for (int i = 0; i < 10; i++) {
        x = i * 10;
        y = i * 10;
        szName = "point_" + std::to_string(i);

        // To write a feature to disk, we must create a local OGRFeature, set attributes and attach geometry before trying to write it to the layer.
        // It is imperative that this feature be instantiated from the OGRFeatureDefn associated with the layer it will be written to.
        OGRFeature *poFeature;

        poFeature = OGRFeature::CreateFeature(poLayer->GetLayerDefn());
        poFeature->SetField("Name", szName.c_str());

        // We create a local geometry object, and assign its copy (indirectly) to the feature.
        // The OGRFeature::SetGeometryDirectly() differs from OGRFeature::SetGeometry() in that the direct method gives ownership of the geometry to the feature.
        // This is generally more efficient as it avoids an extra deep object copy of the geometry.
        OGRPoint pt;
        pt.setX(x);
        pt.setY(y);

        poFeature->SetGeometry(&pt);

        // Now we create a feature in the file.
        // The OGRLayer::CreateFeature() does not take ownership of our feature so we clean it up when done with it.
        if (poLayer->CreateFeature(poFeature) != OGRERR_NONE) {
            printf("Failed to create feature in shapefile.\n");
            exit(1);
        }

        OGRFeature::DestroyFeature(poFeature);
    }

    // 5.Finally we need to close down the datasource in order to ensure headers are written out in an orderly way and all resources are recovered.
    GDALClose(poDS);
}


int main() {
    WritingToOGR();
    ReadingFromOGR();
    return 0;
}