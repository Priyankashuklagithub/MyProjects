#Mean precipitation trend from 2006 to 2100 using NASA NEX GDDP Dataset
var ROI = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.or(ee.Filter.eq('country_na', 'Iraq')));
Map.centerObject(ROI);
Map.addLayer(ROI, {}, 'ROI');

var collection = ee.ImageCollection('NASA/NEX-GDDP');

// Select scenario
var scenario = 'rcp85';

// Filter the collection by date and scenario
var filteredCollection = collection
  .filterDate('2006-01-01', '2100-01-01')//includes one day before the end date
  .filter(ee.Filter.eq('scenario', scenario))
  .select(['pr'])
    .map(function (image) {
    return image.clip(ROI);
  });


print(filteredCollection.limit(20))  
print('Filtered Collection Size:', filteredCollection.size());

var scale = 25000;

var step = 1; // years
var years = ee.List.sequence(2006, 2099, step);

// map over years and then reduce over countries
var output = years.map(function (year) {
  var begin = ee.Date.fromYMD(year, 1, 1);
  var end = begin.advance(step, 'year');

  var collection2 = filteredCollection
      .filterDate(begin, end);


  //The amount in 1 kg m-2 is equivalent to 1 mm 
  var sumImage = collection2
    .sum() // Calculate the annual sum
    .multiply(60*60*24)
    .divide(21)

  // Add metadata properties to the image
  sumImage = sumImage.set({
    'year_begin': begin.format('YYYY'),
    'year_end': end.format('YYYY')
  });

  return sumImage;
});

// Convert the output to an ImageCollection
output = ee.ImageCollection(output);
print(output.limit(1));


var outImage = output.first().select(['pr']);

Map.addLayer(outImage.clip(ROI), {
  min: 0, // Set the minimum value for visualization
  max: 10, // Set the maximum value for visualization
  palette: ['blue', 'purple', 'cyan', 'green', 'yellow', 'red'] // Set a custom color palette
}, 'mean out Image (Temperature Max)');

var numImages = output.size();
print("Number of Images:", numImages);

var addTimeBand = function(image) {
  var timeStart = ee.Number.parse(image.get('year_begin')).toInt();
  var timeImage = ee.Image(timeStart).toInt();
  var img = image.addBands(timeImage.rename('constant'));
  return img;
};

var collectionWithTime = output.map(addTimeBand);

print(collectionWithTime, 'Time band added');

var linearFit = ee.Image(collectionWithTime.select(['constant', 'pr']).reduce(ee.Reducer.linearFit()));
print(linearFit, 'Linear fit image');

// Get the scale (slope) and offset (intercept) coefficients
var scale = linearFit.select('scale');
var offset = linearFit.select('offset');

// Print the scale (slope) and offset (intercept)
print('Scale (Slope):', scale);
print('Offset (Intercept):', offset);

Export.image.toDrive({
  image: linearFit.clip(ROI),
  description: "lineartrend_Final_pr_2006to2100",
  region: ROI,
  scale: 25000, 
})

Map.addLayer(scale.clip(ROI), {
  min: 0, // Set the minimum value for visualization
  max: 10, // Set the maximum value for visualization
  palette: ['blue', 'purple', 'cyan', 'green', 'yellow', 'red'] // Set a custom color palette
}, 'scale');

Map.addLayer(offset.clip(ROI), {
  min: 0, // Set the minimum value for visualization
  max: 10, // Set the maximum value for visualization
  palette: ['blue', 'purple', 'cyan', 'green', 'yellow', 'red'] // Set a custom color palette
}, 'offset');
