# gdal_tutorial

### 1.convert shp to tiff: -- [reference](https://gdal.org/programs/gdal_rasterize.html)

`$ gdal_rasterize -burn 0 -burn 0 -burn 255 -ot Byte -ts 512 512 -l shanghai shanghai.shp shanghai.tif`

### 2.convert shp to GPKG: -- [reference](https://gdal.org/programs/ogr2ogr.html)

`$ ogr2ogr -f GPKG shanghai.gpkg shanghai.shp`

### 3.convert GPKG to shp: -- [reference](https://morphocode.com/using-ogr2ogr-convert-data-formats-geojson-postgis-esri-geodatabase-shapefiles/)

`$ ogr2ogr -f "ESRI Shapefile" shanghai.shp shanghai.gpkg`

### 4.Clip input layer with a bounding box:

`$ ogr2ogr -spat -13.931 34.886 46.23 74.12 -f GPKG shanghai_clip.gpkg shanghai.gpkg`

`$ ogr2ogr -spat 0 0 50 70 -f "ESRI Shapefile" point_out_org.shp point_out.shp`

### 5.Output file format name. Starting with GDAL 2.3, if not specified, the format is guessed from the extension (previously was ESRI Shapefile).

`$ ogr2ogr -f GPKG output.gpkg input.shp`

`$ ogr2ogr -f GeoJSON point.geojson point.shp`

### 6.About polygon -- [reference](http://esri.github.io/geometry-api-java/doc/Polygon.html) -- [reference](https://github.com/Esri/geometry-api-java/wiki)


### 7. Coordinates transform.
   In this situation you will need to find out what coordinate system your data is in. If the data has a coordinate system assigned you can issue a command like that below to convert the data (note that the destination file is specified before the source file):

`$ ogr2ogr -t_srs EPSG:4326 new_datafile.shp datafile.shp`

   If your data set does not have a coordinate system assigned to it but you have found out what it is you can specify the source coordinate system on the command line with the parameter -s_srs, for example:

`$ ogr2ogr -s_srs EPSG:27700 -t_srs EPSG:4326 new_datafile.shp datafile.shp`

   After that, you will have to reproject your file to EPSG:4326 (latitudes and longitudes):

`$ ogr2ogr -t_srs EPSG:4326 nyu_2451_34498_4326.shp nyu_2451_34498_3857.shp`
