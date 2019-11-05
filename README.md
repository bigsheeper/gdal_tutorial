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
