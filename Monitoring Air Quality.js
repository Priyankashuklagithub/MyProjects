#plot SO2 variations in Imphal region
#select the required points for the specific class to monitor
#select the required geometry

var SO2_coll = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_SO2")
                .filter(ee.Filter.or(
                  // ee.Filter.date('2016-01-01','2016-12-31'),
                  ee.Filter.date('2018-01-01','2021-12-31')
                  )).filterBounds(roi).select('SO2_column_number_density');
print(SO2_coll);

var SO2_deg_coll = function(img){
    var SO2_deg = img.select('SO2_column_number_density').multiply(10000000000).clip(roi);
  return SO2_deg.copyProperties(img,['system:index','system:time_end','system:time_start'])
}
var SO2_coll_final = SO2_coll.map(SO2_deg_coll);
print(SO2_coll_final);

var SO2_bands = SO2_coll_final.toBands();
var SO2_stats = SO2_bands.reduceRegions({
  collection:roi,
  reducer: ee.Reducer.mean(),
  scale:1000
});
 Export.table.toDrive({
    description:'SO2-stats settlement', 
    collection: SO2_stats.select(['.*'],null,false), 
    fileFormat: 'CSV'});
var SO2_img = SO2_coll_final.select('SO2_column_number_density').filter(ee.Filter.eq('system:index','2022_04_07'))
var SO2_img1 = SO2_coll_final.select('SO2_column_number_density').filter(ee.Filter.eq('system:index','2020_04_06'))

 var viz = {
  min: -13.15,
  max: 56.85,
  palette: [
    '040274', '040281', '0502a3', '0502b8', '0502ce', '0502e6',
    '0602ff', '235cb1', '307ef3', '269db1', '30c8e2', '32d3ef',
    '3be285', '3ff38f', '86e26f', '3ae237', 'b5e22e', 'd6e21f',
    'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d',
    'ff0000', 'de0101', 'c21301', 'a71001', '911003'
  ],
};

Map.addLayer(SO2_img,viz,'SO2_2022_04_07');
Map.addLayer(SO2_img1,viz,'SO2_2020_04_06');

var chart_SO2 = ui.Chart.image.doySeriesByYear({

  imageCollection:SO2_coll_final, 
  bandName:'SO2_column_number_density', 
  region:class7, 
  regionReducer: ee.Reducer.mean(), 
  scale:1000, 
  sameDayReducer: ee.Reducer.mean(), 
  startDay:1, 
  endDay:365
  }).setOptions({
      interpolateNulls: true,
      lineWidth: 1,
      pointSize: 3,
      fontSize: 32,
      title: 'SO2 settlement',
      vAxis: {title: 'SO2(mol/m^2) settlement'},
      hAxis: {title: 'Day of year', gridlines: {count: 12}}});
  
print(chart_SO2)


#Like wise use different products and classes to monitor CO, HCHO, SO2, NO2, Ch4, Aerosol Index
#COPERNICUS/S5P/NRTI/L3_HCHO
#COPERNICUS/S5P/NRTI/L3_NO2
#COPERNICUS/S5P/NRTI/L3_CO
#COPERNICUS/S5P/OFFL/L3_CH4
#COPERNICUS/S5P/NRTI/L3_AER_AI




















