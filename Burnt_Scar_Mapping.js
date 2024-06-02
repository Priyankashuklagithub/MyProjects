//===========================================================================================
//             BURN SEVERITY MAPPING USING THE NORMALIZED BURN RATIO (NBR)
//===========================================================================================
// Set start and end dates of a period BEFORE the fire. Make sure it is long enough for 
// Sentinel-2 to acquire an image (repitition rate = 5 days). Adjust these parameters, if
// your ImageCollections (see Console) do not contain any elements.
var prefire_start = '2021-02-08';   
var prefire_end = '2021-02-20';

// Now set the same parameters for AFTER the fire.
var postfire_start = '2021-04-20';
var postfire_end = '2021-04-30';

//*******************************************************************************************
//                            SELECT A SATELLITE PLATFORM

// Select remote sensing imagery from two availible satellite sensors. 

// Landsat 8                             |  Sentinel-2 (A&B)

var platform = 'S2';               // <--- assign your choice to the platform variable

//*******************************************************************************************
//*******************************************************************************************


//---------------------------------- Inputs --------------------------------

// Print Satellite platform and dates to console
if (platform == 'S2' | platform == 's2') {
  var ImCol = 'COPERNICUS/S2';
  var pl = 'Sentinel-2';
} else {
  var ImCol = 'LANDSAT/LC08/C01/T1_SR';
  var pl = 'Landsat 8';
}

// Location
var area = ee.FeatureCollection(geometry1);

// Set study area as map center.
//Map.centerObject(area);

//----------------------- Select Landsat imagery by time and location -----------------------

var imagery = ee.ImageCollection(ImCol);

// In the following lines imagery will be collected in an ImageCollection, depending on the
// location of our study area, a given time frame and the ratio of cloud cover.
var prefireImCol = ee.ImageCollection(imagery
    // Filter by dates.
    .filterDate(prefire_start, prefire_end)
    // Filter by location.
    .filterBounds(area));
    
// Select all images that overlap with the study area from a given time frame 
// As a post-fire state we select the 25th of February 2017
var postfireImCol = ee.ImageCollection(imagery
    // Filter by dates.
    .filterDate(postfire_start, postfire_end)
    // Filter by location.
    .filterBounds(area));

// Add the clipped images to the console on the right
print("Pre-fire Image Collection: ", prefireImCol); 
print("Post-fire Image Collection: ", postfireImCol);

//------------------------------- Apply a cloud and snow mask -------------------------------

// Function to mask clouds from the pixel quality band of Sentinel-2 SR data.
function maskS2sr(image) {
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = ee.Number(2).pow(10).int();
  var cirrusBitMask = ee.Number(2).pow(11).int();
  // Get the pixel QA band.
  var qa = image.select('QA60');
  // All flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  // Return the masked image, scaled to TOA reflectance, without the QA bands.
  return image.updateMask(mask)
      .copyProperties(image, ["system:time_start"]);
}

// Function to mask clouds from the pixel quality band of Landsat 8 SR data.
function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = 1 << 3;
  var cloudsBitMask = 1 << 5;
  var snowBitMask = 1 << 4;
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // All flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
      .and(qa.bitwiseAnd(cloudsBitMask).eq(0))
      .and(qa.bitwiseAnd(snowBitMask).eq(0));
  // Return the masked image, scaled to TOA reflectance, without the QA bands.
  return image.updateMask(mask)
      .select("B[0-9]*")
      .copyProperties(image, ["system:time_start"]);
}

// Apply platform-specific cloud mask
if (platform == 'S2' | platform == 's2') {
  var prefire_CM_ImCol = prefireImCol.map(maskS2sr);
  var postfire_CM_ImCol = postfireImCol.map(maskS2sr);
} else {
  var prefire_CM_ImCol = prefireImCol.map(maskL8sr);
  var postfire_CM_ImCol = postfireImCol.map(maskL8sr);
}

//----------------------- Mosaic and clip images to study area -----------------------------

// This is especially important, if the collections created above contain more than one image
// (if it is only one, the mosaic() does not affect the imagery).

var pre_mos = prefireImCol.mosaic().clip(area);
var post_mos = postfireImCol.mosaic().clip(area);

var pre_cm_mos = prefire_CM_ImCol.mosaic().clip(area);
var post_cm_mos = postfire_CM_ImCol.mosaic().clip(area);

// Add the clipped images to the console on the right
print("Pre-fire True Color Image: ", pre_mos); 
print("Post-fire True Color Image: ", post_mos);

//------------------ Calculate NBR for pre- and post-fire images ---------------------------

// Apply platform-specific NBR = (NIR-SWIR2) / (NIR+SWIR2)
if (platform == 'S2' | platform == 's2') {
  var preNBR = pre_cm_mos.normalizedDifference(['B8', 'B12']);
  var postNBR = post_cm_mos.normalizedDifference(['B8', 'B12']);
} else {
  var preNBR = pre_cm_mos.normalizedDifference(['B5', 'B7']);
  var postNBR = post_cm_mos.normalizedDifference(['B5', 'B7']);
}


// Add the NBR images to the console on the right
print("Pre-fire Normalized Burn Ratio: ", preNBR); 
print("Post-fire Normalized Burn Ratio: ", postNBR);

//------------------ Calculate difference between pre- and post-fire images ----------------

// The result is called delta NBR or dNBR
var dNBR_unscaled = preNBR.subtract(postNBR);

// var fireIndices = fireIndices.addBands(preNBR);

// Scale product to USGS standards
var dNBR = dNBR_unscaled.multiply(1000).rename('dnbr').toFloat() ;
print("iiiiiPre-fire Normalized Burn Ratio: ", dNBR); 

// calculate RBR  
var RBR= dNBR.divide(preNBR.add(1.001))
            .rename('rbr').toFloat()
            
   
var RDNBR = dNBR.divide((preNBR.abs()).sqrt()).rename('rdnbr').toFloat()


// Add the difference image to the console on the right
print("RBR: ", RBR);
print("RDNBR: ", RDNBR);

//==========================================================================================
//                                    ADD LAYERS TO MAP

// Add boundary.
Map.addLayer(area.draw({color: 'ffffff', strokeWidth: 5}), {},'Study Area');

//---------------------------------- True Color Imagery ------------------------------------

// Apply platform-specific visualization parameters for true color images
if (platform == 'S2' | platform == 's2') {
  var vis = {bands: ['B4', 'B3', 'B2'], max: 2000, gamma: 1.5};
} else {
  var vis = {bands: ['B4', 'B3', 'B2'], min: 0, max: 4000, gamma: 1.5};
}

// Add the true color images to the map.
Map.addLayer(pre_mos, vis,'Pre-fire image');
Map.addLayer(post_mos, vis,'Post-fire image');

// Add the true color images to the map.
Map.addLayer(pre_cm_mos, vis,'Pre-fire True Color Image - Clouds masked');
Map.addLayer(post_cm_mos, vis,'Post-fire True Color Image - Clouds masked');

//--------------------------- Burn Ratio Product - Greyscale -------------------------------

var grey = ['white', 'black'];
Map.addLayer(dNBR, {min: -1000, max: 1000, palette: grey}, 'dNBR greyscale');

//==========================================================================================
//  CALCULATED BURNED AREA STATISTICS using ArcMap after exporting the images
//==========================================================================================

Export.image.toDrive({image: dNBR, scale: 30, description: "dNBR-grey scale", fileNamePrefix: 'dNBR-greyscale',
  region: area, maxPixels: 1e10,
  crs: 'EPSG: 32646'
  //WGS 84 UTM Zone 46
  });
  
Export.image.toDrive({image: dNBR, scale: 30, description: "DNBR", fileNamePrefix: 'dNBR',
  region: area, maxPixels: 1e10,
  crs: 'EPSG: 32646'
  //WGS 84 UTM Zone 46
  });

Export.image.toDrive({image: RBR, scale: 30, description: "RBR", fileNamePrefix: 'RBR',
  region: area, maxPixels: 1e10,
  crs: 'EPSG: 32646'
});

Export.image.toDrive({image: RDNBR,folder:'GEE/BAM', scale: 30, description: "RDNBR", fileNamePrefix: 'RDNBR',
  region: area, maxPixels: 1e10,crs: 'EPSG: 32646'});
  
Export.image.toDrive({image: preNBR, scale: 30, description: "preNBR", fileNamePrefix: 'preNBR',
  region: area, maxPixels: 1e10,crs: 'EPSG: 32646'});

Export.image.toDrive({image: postNBR, scale: 30, description: "postNBR", fileNamePrefix: 'postNBR',
  region: area, maxPixels: 1e10,crs: 'EPSG: 32646'});
  
Export.image.toDrive({image: (pre_mos.select('B8','B4','B3')).toFloat(), scale: 30, description: "pre fire image", fileNamePrefix: 'Pre fire image',
  region: area, maxPixels: 1e10,crs: 'EPSG: 32646'});
  
Export.image.toDrive({image: (post_mos.select('B8','B4','B3')).toFloat(), scale: 30, description: "post fire image", fileNamePrefix: 'Post fire image',
  region: area, maxPixels: 1e10,crs: 'EPSG: 32646'});
  
Export.image.toDrive({image:( pre_cm_mos.select('B8','B4','B3')).toFloat(), scale: 30, description: "pre cm fire image", fileNamePrefix: 'Pre cm fire image',
  region: area, maxPixels: 1e10,crs: 'EPSG: 32646'});
  
Export.image.toDrive({image: (post_cm_mos.select('B8','B4','B3')).toFloat(), scale: 30, description: "post cm fire image", fileNamePrefix: 'Post cm fire image',
  region: area, maxPixels: 1e10,crs: 'EPSG: 32646'});
// Downloads will be availible in the 'Tasks'-tab on the right.


