Map.addLayer(roi)
Map.setCenter(93.7893, 24.5535,20) ;

// //Load Sentinel-1 C-band SAR Ground Range collection (log scale, VV, descending)
var collectionVV1 = ee.ImageCollection('COPERNICUS/S1_GRD')
.filter(ee.Filter.eq('instrumentMode', 'IW'))
.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
.filterMetadata('resolution_meters', 'equals' , 10)
.filterBounds(roi)
.select('VV')
print(collectionVV, 'Collection VV'); 

function clp(img) {
  return img.clip(roi)
};

var collectionVV = collectionVV1.map(clp)
print(collectionVV)

// Load Sentinel-1 C-band SAR Ground Range collection (log scale, VH, descending)
var collectionVH1 = ee.ImageCollection('COPERNICUS/S1_GRD')
.filter(ee.Filter.eq('instrumentMode', 'IW'))
.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
.filterMetadata('resolution_meters', 'equals' , 10)
.filterBounds(roi)
.select('VH');
print(collectionVH, 'Collection VH');

function clp(img) {
  return img.clip(roi)
};

var collectionVH = collectionVH1.map(clp)
print(collectionVH)

////Filter by date
var SARVV = collectionVV.filterDate('2021-02-01', '2021-04-30').mosaic();
var SARVH = collectionVH.filterDate('2021-02-01', '2021-04-30').mosaic();

// Add the SAR images to "layers" in order to display them
Map.centerObject(roi, 7);
Map.addLayer(SARVV, {min:-15,max:0}, 'SAR VV', 0);
Map.addLayer(SARVH, {min:-25,max:0}, 'SAR VH', 0);

// Function to cloud mask from the pixel_qa band of Landsat 8 SR data.
function maskL8sr(image) {
// Bits 3 and 5 are cloud shadows and clouds, respectively.
var cloudShadowBitMask = 1 << 3;
var cloudsBitMask = 1 << 5;
// Get the pixel QA band.
var qa = image.select('pixel_qa');
// Both flags should be set to zero, indicating clear conditions.
var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
.and(qa.bitwiseAnd(cloudsBitMask).eq(0));
// Return the masked image, scaled to reflectance, without the QA bands.
 return image.updateMask(mask).divide(10000)
.select("B[0-8]*")
.copyProperties(image, ["system:time_start"]);
}

// Extract the images from the Landsat8 collection
var collectionl81 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
.filterDate('2021-02-01', '2021-04-30')
.filterBounds(roi)
.map(maskL8sr);
print(collectionl8, 'Landsat');

function clp(img) {
  return img.clip(roi)
};

var collectionl8 = collectionl81.map(clp)
print(collectionVH)


//Calculate NDVI and create an image that contains all Landsat 8 bands and NDVI
var comp = collectionl8.mean();
var ndvi = comp.normalizedDifference(['B5', 'B4']).rename('NDVI');
var composite = ee.Image.cat(comp,ndvi);

// Add images to layers in order to display them
Map.centerObject(roi, 7);
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.2}, 'Optical');

//Apply filter to reduce speckle
var SMOOTHING_RADIUS = 50;
var SARVV_filtered = SARVV.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters');
var SARVH_filtered = SARVH.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters');

//Display the SAR filtered images
Map.addLayer(SARVV_filtered, {min:-15,max:0}, 'SAR VV Filtered',0);
Map.addLayer(SARVH_filtered, {min:-25,max:0}, 'SAR VH Filtered',0);

//Merge Feature Collections
var newfc = waterbody.merge(phumdis).merge(settlement).merge(horticulture_farm).merge(paddy).merge(vegetation).merge(bare);

// //------------------------------------------------------------------------------
var bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','NDVI'];

//Define the SAR bands to train your data
var final = ee.Image.cat(SARVV_filtered,SARVH_filtered);
var bands = ['VH','VV'];
var training = final.select(bands).sampleRegions({
  collection: newfc,
  properties: ['landcover'],
  scale: 30 }).randomColumn();
  
//Randomly split the samples to set some aside for testing the model's accuracy
//using the "random" column. Roughly 80% for training, 20% for testing.
var split = 0.8;
var training1 = training.filter(ee.Filter.lt('random', split));
var testing = training.filter(ee.Filter.gte('random', split));

//Print these variables to see how much training and testing data you are using
print('Samples n =', training.aggregate_count('.all'));
print('Training n =', training1.aggregate_count('.all'));
print('Testing n =', testing.aggregate_count('.all'));


//Train the classifier
var classifier = ee.Classifier.smileRandomForest(300).train({
  features: training1,
  classProperty: 'landcover',
  inputProperties: bands
});

//Run the Classification
var classified = final.select(bands).classify(classifier);

//Display the Classification
Map.addLayer(classified, 
{min: 1, max: 7, palette: ['137deb', 'c86dd6', 'ff5116', 'ebfe1a',  '9bff63','118112','787d3d']},
'SAR Classification');

// Create a confusion matrix representing resubstitution accuracy.
print('RF- SAR error matrix: ', classifier.confusionMatrix());
print('RF- SAR accuracy: ', classifier.confusionMatrix().accuracy());

// //-------------------------------------------------------------------------
// //SAR-------------------------------------------------------------------------
var validation = testing.classify(classifier);
var testAccuracy = validation.errorMatrix('landcover', 'classification');
print('Validation Error Matrix RF: ', testAccuracy);
print('Validation Overall Accuracy RF: ', testAccuracy.accuracy());
var kappa1 = testAccuracy.kappa();
print('Validation Kappa', kappa1);
// //-------------------------------------------------------------------------

print(composite)
//Repeat for Landsat
//Define the Landsat bands to train your data
var bandsl8 = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','NDVI' ];
var trainingl8 = composite.select(bandsl8).sampleRegions({
  collection: newfc,
  properties: ['landcover'],
  scale: 30
}).randomColumn();
print(trainingl8)

var split = 0.8;
var trainingl81 = trainingl8.filter(ee.Filter.lt('random', split));
var testingl8 = trainingl8.filter(ee.Filter.gte('random', split));
print(trainingl81)
print(testingl8)

//Train the classifier
var classifierl8 = ee.Classifier.smileRandomForest(300).train({
  features: trainingl81,
  classProperty: 'landcover',
  inputProperties: bandsl8
});

//Run the Classification
var classifiedl8 = composite.select(bandsl8).classify(classifierl8);

var validation1 = testingl8.classify(classifierl8);
var testAccuracy1 = validation1.errorMatrix('landcover', 'classification');
print('Validation Error Matrix RF: ', testAccuracy1);
print('Validation Overall Accuracy RF: ', testAccuracy1.accuracy());
var kappa2 = testAccuracy1.kappa();
print('Validation Kappa', kappa2);

//Display the Classification
Map.addLayer(classifiedl8, 
{min: 1, max: 7, palette: ['137deb', 'c86dd6', 'ff5116', 'ebfe1a',  '9bff63','118112','787d3d']},
'Optical Classification');

//// Create a confusion matrix representing resubstitution accuracy.
print('RF-L8 error matrix: ', classifierl8.confusionMatrix());
print('RF-L8 accuracy: ', classifierl8.confusionMatrix().accuracy());


//Define both optical and SAR to train your data
var opt_sar = ee.Image.cat(composite, SARVV_filtered,SARVH_filtered);
var bands_opt_sar = ['VH','VV','B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','NDVI'];
var training_opt_sar = opt_sar.select(bands_opt_sar).sampleRegions({
  collection: newfc,
  properties: ['landcover'],
  scale: 30 }).randomColumn();
  
print(training_opt_sar)

var split = 0.8;
var training_opt_sar1 = training_opt_sar.filter(ee.Filter.lt('random', split));
var testing_opt_sar1 = training_opt_sar.filter(ee.Filter.gte('random', split));
print(training_opt_sar1)
print(testing_opt_sar1)

//Train the classifier
var classifier_opt_sar = ee.Classifier.smileRandomForest(300).train({
  features: training_opt_sar1, 
  classProperty: 'landcover',
  inputProperties: bands_opt_sar 
});

//Run the Classification
var classifiedboth = opt_sar.select(bands_opt_sar).classify(classifier_opt_sar);

var validation2 = testing_opt_sar1.classify(classifier_opt_sar);
var testAccuracy2 = validation2.errorMatrix('landcover', 'classification');
print('Validation Error Matrix RF: ', testAccuracy2);
print('Validation Overall Accuracy RF: ', testAccuracy2.accuracy());
var kappa3 = testAccuracy2.kappa();
print('Validation Kappa', kappa3);

//Display the Classification
var mask_o = composite.select(0).neq(1000)
var mask_r = SARVV_filtered.neq(1000)
var mask = mask_r.updateMask(mask_o)
Map.addLayer(classifiedboth.updateMask(mask), 
{min: 1, max: 7, palette: ['137deb', 'c86dd6', 'ff5116', 'ebfe1a',  '9bff63','118112','787d3d']},
'Optical/SAR Classification');

// Create a confusion matrix representing resubstitution accuracy.
print('RF-Opt/SAR error matrix: ', classifier_opt_sar.confusionMatrix());
print('RF-Opt/SAR accuracy: ', classifier_opt_sar.confusionMatrix().accuracy());

// Export the image, specifying scale and region.
 Export.image.toDrive({
  image: classifiedboth,
  description: "Optical_Radar_2021",
  region: roi,
  crs: 'EPSG: 32646', //WGS 84 UTM Zone 46
  scale: 30,
  maxPixels: 1000000000
})

Export.image.toDrive({
  image: classifiedl8,
  description: "Optical_2021",
  region: roi,
  crs: 'EPSG: 32646', //WGS 84 UTM Zone 46
  scale: 30,
  maxPixels: 1000000000
})
