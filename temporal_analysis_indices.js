
/**
 * Function to mask clouds based on the pixel_qa band of Landsat 8 SR data.
 * @param {ee.Image} image input Landsat 8 SR image
 * @return {ee.Image} cloudmasked Landsat 8 image
 */
function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask);
}

//STUDY AREA
var Mizoram = 
    ee.Geometry.Polygon(
        [[[92.26154447735802, 24.535131817489376],
          [92.26154447735802, 21.800819377722572],
          [93.48102689923302, 21.800819377722572],
          [93.48102689923302, 24.535131817489376]]], null, false);
          

var Landsat = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').select('B2','B3','B4','B5','B6', 'B7', 'B10','pixel_qa');
 var rgbVis = {
  min: 0.0,
  max: 3000,
  bands: ['B4', 'B3', 'B2'], 
}
var area = ee.FeatureCollection(Mizoram);
Map.centerObject(area);

//FILTER THE REQUIRED IMAGES
var SRcollectionMasked =  Landsat
                .filter(ee.Filter.date('2016-01-01','2021-06-30'))
                .map(maskL8sr)
                .filter(ee.Filter.bounds(Mizoram))
                .map(function(image){return image.clip(Mizoram)});

Map.addLayer(SRcollectionMasked, rgbVis,"Masked");
print(SRcollectionMasked,"masked images");
Map.setCenter(92.82734037579552, 23.343365646923534, 7);


// MOSAICKING BY DATE
function mosaicByDate(imcol){
  // imcol: An image collection
  // returns: An image collection
  var imlist = imcol.toList(imcol.size())

  var unique_dates = imlist.map(function(im){
    return ee.Image(im).date().format("YYYY-MM-dd")
  }).distinct()

  var mosaic_imlist = unique_dates.map(function(d){
    d = ee.Date(d)

    var im = imcol
      .filterDate(d, d.advance(1, "day"))
      .mosaic()

    return im.set(
        "system:time_start", d.millis(), 
        "system:id", d.format("YYYY-MM-dd"))
  })

  return ee.ImageCollection(mosaic_imlist)
}

print(mosaicByDate(SRcollectionMasked))

var col = mosaicByDate(SRcollectionMasked)

// Functions to Calculate Indices
function addGVMI(image) {
  var gvmi = image.expression(
  '((NIR+0.1) - (SWIR2 + 0.2)) / ((NIR+0.1) + (SWIR2 + 0.2))',{
    'NIR': image.select('B5'),
    'SWIR2': image.select('B7')
  });
  return image.addBands(gvmi.rename('GVMI'))
}
function addNBR(image) {
  var nbr = image.normalizedDifference(['B5', 'B7']).rename('NBR')
  return image.addBands(nbr)
}
function addNDMI(image) {
  var ndmi = image.normalizedDifference(['B5', 'B6']).rename('NDMI')
  return image.addBands(ndmi)
}
function addAFRI(image) {
  var afri = image.expression(
  '(NIR - (660 * (SWIR1 / ( NIR + (660*SWIR1) ) ))) ',{
    'NIR': image.select('B5'),
    'SWIR1': image.select('B6')
  });
  return image.addBands(afri.rename('AFRI'))
}
function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
  return image.addBands(ndvi)
}

//Functions helping in plotting the data
function LBndvi(image) {
  var LB_NDVI = 0.232
  return image.addBands(LB_NDVI)
}
function LBnbr(image) {
  var LB_NBR = 0.285
  return image.addBands(LB_NBR)
}
function LBndmi(image) {
  var LB_NDMI = 0.086
  return image.addBands(LB_NDMI)
}
function LBafri(image) {
  var LB_AFRI = 1831
  return image.addBands(LB_AFRI)
}
var withGvmi = col.map(addGVMI).map(addNBR).map(addNDMI).map(addAFRI).map(addNDVI).map(LBndvi).map(LBnbr).map(LBndmi).map(LBafri)


print(withGvmi,"gvmi")
Map.addLayer(withGvmi,{bands: 'GVMI',palette: ['blue', 'white', 'green', 'yellow']},'GVMI');


var options = {
  title: 'Temporal plot for normalized indices of dense forest class 30 random points mean',
  hAxis: {title: 'Date'},
  vAxis: {title: 'Value'},
  lineWidth: 2,
  series: {
      colors: ['#009900','FF0000','0000ff']
}};
print(ui.Chart.image.series(withGvmi.select(['NDVI','NBR','NDMI']), dense,ee.Reducer.mean(),30). setOptions(options));  
 
var options = {
  title: 'Temporal plot for mean of dense forest class points for various Indices',
  hAxis: {title: 'Date'},
  vAxis: {title: 'Value'},
  lineWidth: 2,
  series: {
      colors: ['00cc00','FF0000','0000ff','152106']
}};

print(ui.Chart.image.series(withGvmi.select(['NDVI','NBR','NDMI','GVMI','AFRI']), dense,ee.Reducer.mean(),30). setOptions(options));   

var options = {
  title: 'Temporal plot for mean of dense forest class points for various Indices',
  hAxis: {title: 'Date'},
  vAxis: {title: 'Value'},
  lineWidth: 2,
  series: {
      colors: ['00cc00','FF0000','0000ff','152106']
}};

print(ui.Chart.image.series(withGvmi.select(['NDVI','NBR','NDMI','GVMI','AFRI']), dense,ee.Reducer.mean(),30). setOptions(options));   

// dense is the collection of points
var chart = ui.Chart.image.series({
                                   imageCollection:  withGvmi.select(['AFRI','NDVI','NBR','NDMI','constant','constant_1','constant_2','constant_3']),
                                   region: dense ,
                                   scale: 30   ,
                                   reducer:ee.Reducer.mean(),
                                  
                                  })
                          .setSeriesNames(['AFRI(In thousands)','NBR[-1,1]','NDMI[-1,1]','NDVI[-1,1]','LB_NDVI','LB_NBR','LB_NDMI','LB_AFRI'])
                          .setOptions({
                            title: 'Temporal Analysis(2016-2021)-Dense Forest Region' ,
                            series: {
                                     
                                     0: {
                                         targetAxisIndex: 1 ,
                                        
                                         type: "line" ,
                                         lineWidth: 2 ,
                                         pointSize: 0 ,
                                         color: "brown" 
                                        } ,
                                     1: {
                                         targetAxisIndex: 0 ,
                                         type: "line" ,
                                         lineWidth: 2 ,
                                         pointSize: 0 ,
                                         color: "orange" 
                                        } ,
                                    2: {
                                         targetAxisIndex: 0 ,
                                         type: "line" ,
                                         lineWidth: 2 ,
                                         pointSize: 0 ,
                                         color: "blue" 
                                        } ,
                                   3: {
                                         targetAxisIndex: 0 ,
                                         type: "line" ,
                                         lineWidth: 2 ,
                                         pointSize: 0 ,
                                         color: "green" 
                                        } ,
                                  4: {
                                         targetAxisIndex: 0 ,
                                         type: "line" ,
                                         lineDashStyle: [4, 4],
                                         lineWidth: 2 ,
                                         pointSize: 0 ,
                                         color: "green" 
                                        }, 
                                  5: {
                                         targetAxisIndex: 0 ,
                                         type: "line" ,
                                         lineDashStyle: [4, 4],
                                         lineWidth: 2 ,
                                         pointSize: 0 ,
                                         color: "orange" 
                                        }, 
                                  6: {
                                         targetAxisIndex: 0 ,
                                         type: "line" ,
                                         lineDashStyle: [4, 4],
                                         lineWidth: 2 ,
                                         pointSize: 0 ,
                                         color: "blue" 
                                        },
                                    7: {
                                         targetAxisIndex: 1 ,
                                         type: "line" ,
                                         lineDashStyle: [4, 4],
                                         lineWidth: 2 ,
                                         pointSize: 0 ,
                                         color: "brown" 
                                        } 
                                    } ,
                            hAxis: {
                                    title: 'Date', 
                                    titleTextStyle: { italic: false, bold: true }
                                   } ,
                            vAxis: {
                                title:'Value',
                            
                                    0: {
                                        title: "AFRI" ,
                                        baseline: 0 ,
                                        titleTextStyle: { bold: true , color: 'brown' }
                                      } ,
                                    1: {
                                        title: "NDVI" ,
                                        baseline: 0  ,
                                        titleTextStyle: { bold: true, color: 'green' }
                                      },
                                    2: {
                                        title: "NBR" ,
                                        baseline: 0  ,
                                        titleTextStyle: { bold: true, color: 'orange' }
                                      },
                                    3: {
                                        title: "NDMI" ,
                                        baseline: 0  ,
                                        titleTextStyle: { bold: true, color: 'blue' }
                                      },
                                    4: {
                                        title: "LB_NDVI" ,
                                        baseline: 0  ,
                                        titleTextStyle: { bold: true, color: 'green' }
                                      },
                                    5: {
                                        title: "LB_NBR" ,
                                        baseline: 0  ,
                                        titleTextStyle: { bold: true, color: 'blue' }
                                      },
                                    6: {
                                        title: "LB_NDMI" ,
                                        baseline: 0  ,
                                        titleTextStyle: { bold: true, color: 'orange' }
                                      },
                                   7: {
                                        title: "LB_AFRI" ,
                                        baseline: 0  ,
                                        titleTextStyle: { bold: true, color: 'brown' }
                                      }
                                   } ,
                            curveType: 'function'
                          });

print(chart)
