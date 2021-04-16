#--------------------------------------------------------
## nested CV but with random splits (rather than spatial)
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

# outer_cv=GroupShuffleSplit(n_splits=5, random_state=1, train_size=.80)
# inner_cv=GroupShuffleSplit(n_splits=5, random_state=1, train_size=.80)

outer_cv = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
inner_cv = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)

# lists to store results of CV testing
acc = []
f1 = []
roc_auc = []
i=1
for train_index, test_index in outer_cv.split(X, y, groups=spatial_groups):
    print('working on '+str(i)+'/'+str(5)+' outer cv split', end='\r')
    model = Classifier(random_state=1)

    # index training, testing, and coordinate data
    X_tr, X_tt = X[train_index, :], X[test_index, :]
    y_tr, y_tt = y[train_index], y[test_index]
    coords = coordinates[train_index]
    
    clf = GridSearchCV(estimator=model,
                       param_grid=param_grid,
                       scoring=metric,
                       n_jobs=ncpus,
                       refit=True,
                       cv=inner_cv.split(X_tr,
                                         y_tr,
                                         groups=spatial_groups[train_index]))
    
    clf.fit(X_tr, y_tr)
    #predict using the best model
    best_model = clf.best_estimator_
    pred = best_model.predict(X_tt)

    # evaluate model w/ multiple metrics
    # ROC AUC
    probs = best_model.predict_proba(X_tt)
    probs = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(y_tt, probs)
    auc_ = auc(fpr, tpr)
    roc_auc.append(auc_)
    # Overall accuracy
    ac = balanced_accuracy_score(y_tt, pred)
    acc.append(ac)
    # F1 scores
    f1_ = f1_score(y_tt, pred)
    f1.append(f1_)
    i+=1

#-------------------------------------------------------

#FEATURE SELECTION USING UNIVARIATE METHODS
num_of_features = 12
selection_method = mutual_info_classif

#Select the features
selector = SelectKBest(selection_method, k=num_of_features)
selected_features = selector.fit_transform(model_input[:, model_col_indices], model_input[:, 0])
print(selected_features.shape)

#Print the name of the features selected
idx_of_selections = selector.get_support(indices=True)
selected_columns=list(np.array(column_names[1:])[idx_of_selections])

print('Features selected:')
print(selected_columns)

model_input = np.append(model_input[:,0].reshape(len(model_input[:,0]), 1), selected_features, axis=1)
selected_columns.insert(0, column_names[0])
print(model_input.shape)

#------------------------------------------------------
#LINEAR DISCRIMINANT ANALYSIS FOR FEATURE SELECTION
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Instatiate a standard scalar method
sc = StandardScaler()

#scale features
X = sc.fit_transform(model_input[:, model_col_indices])

#fit LDA model
y=model_input[:, 0]
clf = LinearDiscriminantAnalysis().fit(X,y)

#get the relative frequency of the classes to use as weights for the LDA coefficients.
_, cls_frq_wgts = np.unique(y, return_counts=True)
cls_frq_wgts = cls_frq_wgts / cls_frq_wgts.sum()

# Weight the LDA coefficients by class frequency.
lda_coef = (cls_frq_wgts*clf.coef_.T).T.mean(axis=0)

# The LDA coefficients for the features in descending order.
desc_lda_coef_inds = np.argsort(abs(lda_coef))[::-1]

lda_table = pd.DataFrame({'name': column_names[1:], 'coef': lda_coef})

# Sort the data variable names by the absolute value of the sum of their coefficients.
lda_table['abs_coef'] = abs(lda_table.coef)
lda_table = lda_table.sort_values('abs_coef', ascending=False)

# create a bar chart
lda_table.abs_coef.plot.bar(figsize=(12,6))
plt.xticks(ticks=range(len(lda_table)), labels=lda_table['name'], rotation=70, ha='right')
plt.title('LDA feature importance')
plt.show()

#-----------------------------------------------
#Selecting best features from model
num_of_features = 10

selector=SelectFromModel(
    estimator=ExtraTreesClassifier(), threshold=-np.inf, max_features=num_of_features
)
selected_features = selector.fit_transform(model_input[:, model_col_indices], model_input[:, 0])

#Print the name of the features selected
idx_of_selections = selector.get_support(indices=True)
selected_columns=list(np.array(column_names[1:])[idx_of_selections])

print('Features selected:')
print(selected_columns)

#Recreate our `model_input` and `column_name` parameters, using only the selected features
model_input = np.append(model_input[:,0].reshape(len(model_input[:,0]), 1), selected_features, axis=1)
selected_columns.insert(0, column_names[0])
print(model_input.shape)

#------------------------------------------------

#Point sampling of raster for validation purpose
prediction = rasterio.open(pred_tif)
coords = [(x,y) for x, y in zip(ground_truth.geometry.x, ground_truth.geometry.y)]
# Sample the raster at every point location and store values in DataFrame
ground_truth['Prediction'] = [int(x[0]) for x in prediction.sample(coords)]
ground_truth.head()


#-----------------------------------------
#Clip to southern or northern regions (testing submodels)

clip = gpd.read_file('data/eastern_south.shp')
input_data = gpd.overlay(input_data,clip, how='intersection').reset_index()


#-------------------------------------------
#predicting using annual MADs and pre-computed other feature saved to disk

# load the column_names
with open(training_data, 'r') as file:
    header = file.readline()
    
column_names = header.split()[1:]
 #load data for calculating annual gm+mads
    with HiddenPrints():
        ds = load_ard(dc=dc,
                  products=products,
                  dask_chunks=dask_chunks,
                  **query)

    ds = ds / 10000
    
    #compute annual tmads (requires computing annual gm)
    mads = xr_geomedian_tmad(ds).compute()
    mads['sdev'] = -np.log(mads['sdev'])
    mads['bcdev'] = -np.log(mads['bcdev'])
    mads['edev'] = -np.log(mads['edev'])
    mads = mads[['sdev','bcdev','edev']] #drop gm
    mads.to_netcdf(results+ 'input/annual_mads/Eastern_tile_'+g_id+'_annual_mads.nc')
    mads = mads.chunk(dask_chunks)
    
    #load other data -
    data = xr.open_dataset(results+'input/Eastern_tile_'+g_id+'_inputs.nc').chunk(dask_chunks)
    
    #extract the variables that are common between this model run and previous saved inputs
    xx = data[[f for f in column_names[1:] if f not in ['edev', 'sdev','bcdev']]]
    xxx = xr.merge([xx,mads],compat='override')
    
    #index the arrays in order that matched the column-names in TD
    xxx = xxx[column_names[1:]]

##---------------------------------------------------------------
# Clip S2 tiles geojson to AEZ and plot nicely
import geopandas as gpd
import matplotlib.pyplot as plt

tiles = gpd.read_file('data/gm_s2_2019-2019--P1Y.geojson')
eastern = gpd.read_file('data/Eastern.shp')

if tiles.crs == eastern.crs:
    intersect=gpd.overlay(tiles,eastern, how='intersection')

intersect.plot(figsize=(10,20), linewidth=0.2, facecolor="none", edgecolor="black")
for idx, row in intersect.iterrows():
    plt.annotate(s=row['title'][3:5]+','+row['title'][9:],
                 xy=row['coords'],
                 horizontalalignment='center',
                fontsize=7
                )

    
#----------------------------------------------------
# Outlier detection

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

training_data = "results/training_data/gm_mads_two_seasons_training_data_20210203.txt"
outlier_percent = 0.1

# load the data
X = np.loadtxt(training_data)

# load the column_names
with open(training_data, 'r') as file:
    header = file.readline()
    
column_names = header.split()[1:]

# Extract relevant indices from training data
model_col_indices = [column_names.index(var_name) for var_name in column_names[1:]]

# separate the crop and non-crop classes so we can run
# outlier detection on classes separately
crop=X[X[:,0]==1]
noncrop=X[X[:,0]==0]

crop_X=crop[:, model_col_indices]
crop_y=crop[:, 0]

noncrop_X=noncrop[:, model_col_indices]
noncrop_y=noncrop[:, 0]

def outlier_detect(X, y, model, **kwargs):

    # summarize the shape of the training dataset
    print("Size before removing outliers: ", X.shape, y.shape)

    # identify outliers in the training dataset
    # ee = OneClassSVM(nu=outlier_percent)
    ee = model(**kwargs)
    yhat = ee.fit_predict(X)

    # select all rows that are not outliers
    mask = yhat != -1
    arr_X, arr_y = X[mask, :], y[mask]

    # summarize the shape of the updated training dataset
    print("Size after removing outliers: ",arr_X.shape, arr_y.shape)
    
    return arr_X, arr_y

#run outlier detection
crop_X, crop_y = outlier_detect(crop_X, crop_y, OneClassSVM, nu=outlier_percent)
noncrop_X, noncrop_y = outlier_detect(noncrop_X, noncrop_y, OneClassSVM, nu=outlier_percent)

# rebuild arrays with class and data
crop=np.insert(crop_X, 0, values=crop_y, axis=1)
noncrop=np.insert(noncrop_X, 0, values=noncrop_y, axis=1)

##-------------------------------------------------------------
# IMAGE SEGMENTATION USING POLYGON ZONAL STATS
#store temp files somewhere
directory=results+'tmp_'
if not os.path.exists(directory):
    os.mkdir(directory)

tmp='tmp_/'

#inputs to image seg
kea_file = results+tif_to_seg[:-4]+'.kea'
segmented_kea_file = results+tif_to_seg[:-4]+'_segmented.kea'

#convert tiff to kea
gdal.Translate(destName=kea_file,
               srcDS=results+tif_to_seg,
               format='KEA',
               outputSRS='EPSG:6933')

#run image seg
print('image segmentation...')
with HiddenPrints():
    segutils.runShepherdSegmentation(inputImg=kea_file,
                                     outputClumps=segmented_kea_file,
                                     tmpath=results+tmp,
                                     numClusters=60,
                                     minPxls=min_seg_size)
    
#open segments
da=xr.open_rasterio(segmented_kea_file).squeeze()

#convert to polygons and export to disk
print('   writing segments to shapefile...')
with HiddenPrints():
    gdf_seg = xr_vectorize(da, attribute_col='attribute')
print("   Number of segments: "+str(len(gdf_seg)))
gdf_seg.to_file(results+tif_to_seg[:-4]+'_segmented.shp')

#calculate zonal-stats
print('zonal statistics...')
zonal_stats_parallel(shp=results+tif_to_seg[:-4]+'_segmented.shp',
       raster=results+pred_tif,
       statistics=['majority'],
       out_shp=results+tmp+"zonal_stats.shp",
       ncpus=ncpus if ncpus<25 else 25, #otherwise too much mem consumption
       nodata=-1
           )

#rasterize the zonal-stats
with HiddenPrints():
    gdf_zs=gpd.read_file(results+tmp+"zonal_stats.shp")
    predict_zs = xr_rasterize(gdf_zs, da, attribute_col='majority')

#write to disk
write_cog(predict_zs, results+ 'prediction_object.tif', overwrite=True)

#remove the tmp folder
shutil.rmtree(results+tmp)
os.remove(kea_file)
os.remove(segmented_kea_file)


###############################################
##------FEATURE LAYER CUTS---------------------
###############################################
"""
This script exists simply to keep the notebooks
'Extract_training_data.ipynb' and 'Predict.ipynb' tidy by
not clutering them up with custom training data functions.
"""

import pyproj
import dask
import hdstats
import datacube
import numpy as np
import sys
import xarray as xr
import warnings
import dask.array as da
from datacube.utils.geometry import assign_crs
from datacube.testutils.io import rio_slurp_xarray
from odc.algo import randomize, reshape_for_geomedian, xr_reproject, xr_geomedian
from odc.algo._dask import reshape_yxbt

sys.path.append('../../Scripts')
from deafrica_bandindices import calculate_indices
from deafrica_temporal_statistics import xr_phenology, temporal_statistics
from deafrica_classificationtools import HiddenPrints
from deafrica_datahandling import load_ard

warnings.filterwarnings("ignore")

def gm_mads_two_seasons_production(ds1, ds2):
    """
    Feature layer function for production run of
    eastern crop-mask
    """            
    def fun(ds, era):
        #normalise SR and edev bands
        for band in ds.data_vars:
            if band not in ['sdev', 'bcdev']:
                ds[band] = ds[band] / 10000
                
        gm_mads = calculate_indices(ds,
                               index=['NDVI','LAI','MNDWI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        gm_mads['sdev'] = -np.log(gm_mads['sdev'])
        gm_mads['bcdev'] = -np.log(gm_mads['bcdev'])
        gm_mads['edev'] = -np.log(gm_mads['edev'])
        
        #rainfall climatology
        if era == '_S1':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S2':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc'),  crs='epsg:4326')
        
        chirps = xr_reproject(chirps,ds.geobox,"bilinear")
        gm_mads['rain'] = chirps
        
        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band:band+era})
        
        return gm_mads
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds1.geobox)
    slope = slope.to_dataset(name='slope')
    
    result = xr.merge([epoch1,
                       epoch2,
                       slope],compat='override')
    
    result = result.astype(np.float32)
    return result.squeeze()

def annual_gm_mads_evi_predict(ds):
    
    dc = datacube.Datacube(app='training')
    dask_chunks={'x':1250,'y':1250}
    # grab gm+tmads
    gm_mads=dc.load(product='ga_s2_gm',
                    time='2019',
                    like=ds.geobox,
                    measurements=['red', 'blue', 'green', 'nir',
                                 'swir_1', 'swir_2', 'red_edge_1',
                                 'red_edge_2', 'red_edge_3', 'SMAD',
                                 'BCMAD','EMAD'],
                    dask_chunks=dask_chunks)
    
    gm_mads['SMAD'] = -np.log(gm_mads['SMAD'])
    gm_mads['BCMAD'] = -np.log(gm_mads['BCMAD'])
    gm_mads['EMAD'] = -np.log(gm_mads['EMAD']/10000)
    
    #calculate band indices on gm
    gm_mads = calculate_indices(gm_mads,
                               index=['EVI','LAI','MNDWI'],
                               drop=False,
                               collection='s2')
    
    #normalise spectral GM bands 0-1
    for band in gm_mads.data_vars:
        if band not in ['SMAD', 'BCMAD','EMAD', 'EVI', 'LAI', 'MNDWI']:
            gm_mads[band] = gm_mads[band] / 10000
    
    gm_mads=gm_mads.persist()
#     gm_mads=gm_mads.chunk(dask_chunks)
    
    #calculate EVI on annual timeseries
    evi = calculate_indices(ds, index=['EVI'], drop=True, normalise=True, collection='s2')
    print('compute evi')
    evi = evi.persist()
#     evi = evi.chunk(dask_chunks)
    print('evi stats')
    gm_mads['evi_std'] = evi.EVI.std(dim='time')
    evi = evi.quantile([0.1,0.25,0.75,0.9], dim='time')
    
    # EVI stats 
    gm_mads['evi_10'] = evi.EVI.sel(quantile='0.1').squeeze().drop('quantile')
    gm_mads['evi_25'] = evi.EVI.sel(quantile='0.25').squeeze().drop('quantile')
    gm_mads['evi_75'] = evi.EVI.sel(quantile='0.75').squeeze().drop('quantile')
    gm_mads['evi_90'] = evi.EVI.sel(quantile='0.90').squeeze().drop('quantile')
    gm_mads['evi_range'] = gm_mads['evi_90'] - gm_mads['evi_10']
    
    #rainfall climatology
    print('ancillary stuff')
    chirps_S1 = xr_reproject(assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc'),
                                        crs='epsg:4326'), ds.geobox,"bilinear")
    
    chirps_S2 = xr_reproject(assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc'), 
                                        crs='epsg:4326'), ds.geobox,"bilinear")
        
    gm_mads['rain_S1'] = chirps_S1.chunk(dask_chunks)
    gm_mads['rain_S2'] = chirps_S2.chunk(dask_chunks)
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope').chunk(dask_chunks)
    
    result = xr.merge([gm_mads,slope],compat='override')

    return result.squeeze()


def annual_gm_mads_evi_training(ds):
    dc = datacube.Datacube(app='training')
    
    # grab gm+tmads
    gm_mads=dc.load(product='ga_s2_gm',time='2019',like=ds.geobox,
                   measurements=['red', 'blue', 'green', 'nir',
                                 'swir_1', 'swir_2', 'red_edge_1',
                                 'red_edge_2', 'red_edge_3', 'SMAD',
                                 'BCMAD','EMAD'])
    
    gm_mads['SMAD'] = -np.log(gm_mads['SMAD'])
    gm_mads['BCMAD'] = -np.log(gm_mads['BCMAD'])
    gm_mads['EMAD'] = -np.log(gm_mads['EMAD']/10000)
    
    #calculate band indices on gm
    gm_mads = calculate_indices(gm_mads,
                               index=['EVI','LAI','MNDWI'],
                               drop=False,
                               collection='s2')
    
    #normalise spectral GM bands 0-1
    for band in gm_mads.data_vars:
        if band not in ['SMAD', 'BCMAD','EMAD', 'EVI', 'LAI', 'MNDWI']:
            gm_mads[band] = gm_mads[band] / 10000
    
    #calculate EVI on annual timeseries
    evi = calculate_indices(ds,index=['EVI'], drop=True, normalise=True, collection='s2')
    
    # EVI stats 
    gm_mads['evi_std'] = evi.EVI.std(dim='time')
    gm_mads['evi_10'] = evi.EVI.quantile(0.1, dim='time')
    gm_mads['evi_25'] = evi.EVI.quantile(0.25, dim='time')
    gm_mads['evi_75'] = evi.EVI.quantile(0.75, dim='time')
    gm_mads['evi_90'] = evi.EVI.quantile(0.9, dim='time')
    gm_mads['evi_range'] = gm_mads['evi_90'] - gm_mads['evi_10']
    
    #rainfall climatology
    chirps_S1 = xr_reproject(assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc'),
                                        crs='epsg:4326'), ds.geobox,"bilinear")
    
    chirps_S2 = xr_reproject(assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc'), 
                                        crs='epsg:4326'), ds.geobox,"bilinear")
        
    gm_mads['rain_S1'] = chirps_S1
    gm_mads['rain_S2'] = chirps_S2
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope')#.chunk({'x':2000,'y':2000})
    
    result = xr.merge([gm_mads,slope],compat='override')

    return result.squeeze()

    

def gm_mads_two_seasons_predict(ds):
    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12'))

    def fun(ds, era):
        #geomedian and tmads
        #gm_mads = xr_geomedian_tmad(ds)
        gm_mads = xr_geomedian_tmad_new(ds).compute()
        gm_mads = calculate_indices(gm_mads,
                               index=['NDVI','LAI','MNDWI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        gm_mads['sdev'] = -np.log(gm_mads['sdev'])
        gm_mads['bcdev'] = -np.log(gm_mads['bcdev'])
        gm_mads['edev'] = -np.log(gm_mads['edev'])
        gm_mads = gm_mads.chunk({'x':2000,'y':2000})
        
        #rainfall climatology
        if era == '_S1':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S2':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc'),  crs='epsg:4326')
        
        chirps = xr_reproject(chirps,ds.geobox,"bilinear")
        chirps = chirps.chunk({'x':2000,'y':2000})
        gm_mads['rain'] = chirps
        
        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band:band+era})
        
        return gm_mads
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope').chunk({'x':2000,'y':2000})
    
    result = xr.merge([epoch1,
                       epoch2,
                       slope],compat='override')

    return result.squeeze()

def gm_mads_two_seasons_training(ds):
    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12')) 

    def fun(ds, era):
        #geomedian and tmads
        gm_mads = xr_geomedian_tmad(ds)
        gm_mads = calculate_indices(gm_mads,
                               index=['NDVI','LAI','MNDWI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        gm_mads['sdev'] = -np.log(gm_mads['sdev'])
        gm_mads['bcdev'] = -np.log(gm_mads['bcdev'])
        gm_mads['edev'] = -np.log(gm_mads['edev'])
        
        #rainfall climatology
        if era == '_S1':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S2':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc'),  crs='epsg:4326')
        
        chirps = xr_reproject(chirps,ds.geobox,"bilinear")
        gm_mads['rain'] = chirps
        
        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band:band+era})
        
        return gm_mads
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope')
    
    result = xr.merge([epoch1,
                       epoch2,
                       slope],compat='override')

    return result.squeeze()

def gm_two_seasons_mads_annual_training(ds):
    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12')) 

    def fun(ds, era):
        #six-month geomedians
        gm_mads = xr_geomedian(ds)
        gm_mads = calculate_indices(gm_mads,
                               index=['NDVI','LAI','MNDWI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        #rainfall climatology
        if era == '_S1':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc'),  crs='epsg:4326')
        if era == '_S2':
            chirps = assign_crs(xr.open_rasterio('/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc'),  crs='epsg:4326')
        
        chirps = xr_reproject(chirps,ds.geobox,"bilinear")
        gm_mads['rain'] = chirps
        
        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band:band+era})
        
        return gm_mads
    
    epoch1 = fun(ds1, era='_S1')
    epoch2 = fun(ds2, era='_S2')
    
    mads = xr_geomedian_tmad(ds)
    mads['sdev'] = -np.log(mads['sdev'])
    mads['bcdev'] = -np.log(mads['bcdev'])
    mads['edev'] = -np.log(mads['edev'])
    mads = mads[['sdev','bcdev','edev']]
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope')
    
    result = xr.merge([epoch1,
                       epoch2,
                       mads,
                       slope],compat='override')

    return result.squeeze()


def gm_mads_evi_rainfall(ds):
    """
    6 monthly and annual 
    gm + mads
    evi stats (10, 50, 90 percentile, range, std)
    rainfall actual stats (min, mean, max, range, std) from monthly data
    rainfall clim stats (min, mean, max, range, std) from monthly data
    """
    dc = datacube.Datacube(app='training')
    ds = ds / 10000
    ds = ds.rename({'nir_1':'nir_wide', 'nir_2':'nir'})
    ds1 = ds.sel(time=slice('2019-01', '2019-06'))
    ds2 = ds.sel(time=slice('2019-07', '2019-12')) 
    
    chirps = []
    chpclim = []
    for m in range(1,13):
        chirps.append(xr_reproject(assign_crs(xr.open_rasterio(f'/g/data/CHIRPS/monthly_2019/chirps-v2.0.2019.{m:02d}.tif').squeeze().expand_dims({'time':[m]}), crs='epsg:4326'), 
                                   ds.geobox, "bilinear"))
        chpclim.append(rio_slurp_xarray(f'https://deafrica-data-dev.s3.amazonaws.com/product-dev/deafrica_chpclim_50n_50s_{m:02d}.tif', gbox=ds.geobox, 
                                        resapling='bilinear').expand_dims({'time':[m]}))
    
    chirps = xr.concat(chirps, dim='time')
    chpclim = xr.concat(chpclim, dim='time')
   
    def fun(ds, chirps, chpclim, era):
        ds = calculate_indices(ds,
                               index=['EVI'],
                               drop=False,
                               normalise=False,
                               collection='s2')        
        #geomedian and tmads
        gm_mads = xr_geomedian_tmad(ds)
        gm_mads = calculate_indices(gm_mads,
                               index=['EVI','NDVI','LAI','MNDWI'],
                               drop=False,
                               normalise=False,
                               collection='s2')
        
        gm_mads['sdev'] = -np.log(gm_mads['sdev'])
        gm_mads['bcdev'] = -np.log(gm_mads['bcdev'])
        gm_mads['edev'] = -np.log(gm_mads['edev'])
        
        # EVI stats 
        gm_mads['evi_10'] = ds.EVI.quantile(0.1, dim='time')
        gm_mads['evi_50'] = ds.EVI.quantile(0.5, dim='time')
        gm_mads['evi_90'] = ds.EVI.quantile(0.9, dim='time')
        gm_mads['evi_range'] = gm_mads['evi_90'] - gm_mads['evi_10']
        gm_mads['evi_std'] = ds.EVI.std(dim='time')

        # rainfall actual
        gm_mads['rain_min'] = chirps.min(dim='time')
        gm_mads['rain_mean'] = chirps.mean(dim='time')
        gm_mads['rain_max'] = chirps.max(dim='time')
        gm_mads['rain_range'] = gm_mads['rain_max'] - gm_mads['rain_min']
        gm_mads['rain_std'] = chirps.std(dim='time')
         
        # rainfall climatology
        gm_mads['rainclim_min'] = chpclim.min(dim='time')
        gm_mads['rainclim_mean'] = chpclim.mean(dim='time')
        gm_mads['rainclim_max'] = chpclim.max(dim='time')
        gm_mads['rainclim_range'] = gm_mads['rainclim_max'] - gm_mads['rainclim_min']
        gm_mads['rainclim_std'] = chpclim.std(dim='time')
                
        for band in gm_mads.data_vars:
            gm_mads = gm_mads.rename({band:band+era})
        
        return gm_mads
    
    epoch0 = fun(ds, chirps, chpclim, era='_S0')
    time, month = slice('2019-01', '2019-06'), slice(1, 6)
    epoch1 = fun(ds.sel(time=time), chirps.sel(time=month), chpclim.sel(time=month), era='_S1')
    time, month = slice('2019-07', '2019-12'), slice(7, 12)
    epoch2 = fun(ds.sel(time=time), chirps.sel(time=month), chpclim.sel(time=month), era='_S2')
    
    #slope
    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    slope = rio_slurp_xarray(url_slope, gbox=ds.geobox)
    slope = slope.to_dataset(name='slope')
    
    result = xr.merge([epoch0,
                       epoch1,
                       epoch2,
                       slope],compat='override')
    
    return result.squeeze()


def xr_geomedian_tmad(ds, axis='time', where=None, **kw):
    """
    :param ds: xr.Dataset|xr.DataArray|numpy array
    Other parameters:
    **kwargs -- passed on to pcm.gnmpcm
       maxiters   : int         1000
       eps        : float       0.0001
       num_threads: int| None   None
    """

    import hdstats
    def gm_tmad(arr, **kw):
        """
        arr: a high dimensional numpy array where the last dimension will be reduced. 
    
        returns: a numpy array with one less dimension than input.
        """
        gm = hdstats.nangeomedian_pcm(arr, **kw)
        nt = kw.pop('num_threads', None)
        emad = hdstats.emad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        smad = hdstats.smad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        bcmad = hdstats.bcmad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        return np.concatenate([gm, emad, smad, bcmad], axis=-1)


    def norm_input(ds, axis):
        if isinstance(ds, xr.DataArray):
            xx = ds
            if len(xx.dims) != 4:
                raise ValueError("Expect 4 dimensions on input: y,x,band,time")
            if axis is not None and xx.dims[3] != axis:
                raise ValueError(f"Can only reduce last dimension, expect: y,x,band,{axis}")
            return None, xx, xx.data
        elif isinstance(ds, xr.Dataset):
            xx = reshape_for_geomedian(ds, axis)
            return ds, xx, xx.data
        else:  # assume numpy or similar
            xx_data = ds
            if xx_data.ndim != 4:
                raise ValueError("Expect 4 dimensions on input: y,x,band,time")
            return None, None, xx_data

    kw.setdefault('nocheck', False)
    kw.setdefault('num_threads', 1)
    kw.setdefault('eps', 1e-6)

    ds, xx, xx_data = norm_input(ds, axis)
    is_dask = dask.is_dask_collection(xx_data)

    if where is not None:
        if is_dask:
            raise NotImplementedError("Dask version doesn't support output masking currently")

        if where.shape != xx_data.shape[:2]:
            raise ValueError("Shape for `where` parameter doesn't match")
        set_nan = ~where
    else:
        set_nan = None

    if is_dask:
        if xx_data.shape[-2:] != xx_data.chunksize[-2:]:
            xx_data = xx_data.rechunk(xx_data.chunksize[:2] + (-1, -1))

        data = da.map_blocks(lambda x: gm_tmad(x, **kw),
                             xx_data,
                             name=randomize('geomedian'),
                             dtype=xx_data.dtype, 
                             chunks=xx_data.chunks[:-2] + (xx_data.chunks[-2][0]+3,),
                             drop_axis=3)
    else:
        data = gm_tmad(xx_data, **kw)

    if set_nan is not None:
        data[set_nan, :] = np.nan

    if xx is None:
        return data

    dims = xx.dims[:-1]
    cc = {k: xx.coords[k] for k in dims}
    cc[dims[-1]] = np.hstack([xx.coords[dims[-1]].values,['edev', 'sdev', 'bcdev']])
    xx_out = xr.DataArray(data, dims=dims, coords=cc)

    if ds is None:
        xx_out.attrs.update(xx.attrs)
        return xx_out

    ds_out = xx_out.to_dataset(dim='band')
    for b in ds.data_vars.keys():
        src, dst = ds[b], ds_out[b]
        dst.attrs.update(src.attrs)

    return assign_crs(ds_out, crs=ds.geobox.crs)


def xr_geomedian_tmad_new(ds, **kw):
    """
    Same as other one but uses reshape_yxbt instead of
    reshape_for_geomedian
    """

    import hdstats
    def gm_tmad(arr, **kw):
        """
        arr: a high dimensional numpy array where the last dimension will be reduced. 
    
        returns: a numpy array with one less dimension than input.
        """
        gm = hdstats.nangeomedian_pcm(arr, **kw)
        nt = kw.pop('num_threads', None)
        emad = hdstats.emad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        smad = hdstats.smad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        bcmad = hdstats.bcmad_pcm(arr, gm, num_threads=nt)[:,:, np.newaxis]
        return np.concatenate([gm, emad, smad, bcmad], axis=-1)


    def norm_input(ds):
        if isinstance(ds, xr.Dataset):
            xx = reshape_yxbt(ds, yx_chunks=500)
            return ds, xx, xx.data

    kw.setdefault('nocheck', False)
    kw.setdefault('num_threads', 1)
    kw.setdefault('eps', 1e-6)

    ds, xx, xx_data = norm_input(ds)
    is_dask = dask.is_dask_collection(xx_data)

    if is_dask:
        data = da.map_blocks(lambda x: gm_tmad(x, **kw),
                             xx_data,
                             name=randomize('geomedian'),
                             dtype=xx_data.dtype, 
                             chunks=xx_data.chunks[:-2] + (xx_data.chunks[-2][0]+3,),
                             drop_axis=3)
    
    dims = xx.dims[:-1]
    cc = {k: xx.coords[k] for k in dims}
    cc[dims[-1]] = np.hstack([xx.coords[dims[-1]].values,['edev', 'sdev', 'bcdev']])
    xx_out = xr.DataArray(data, dims=dims, coords=cc)

    if ds is None:
        xx_out.attrs.update(xx.attrs)
        return xx_out

    ds_out = xx_out.to_dataset(dim='band')
    for b in ds.data_vars.keys():
        src, dst = ds[b], ds_out[b]
        dst.attrs.update(src.attrs)

    return assign_crs(ds_out, crs=ds.geobox.crs)


def merge_tifs_into_ds(
    root_fld: str,
    tifs: List[str],
    rename_dict: Optional[Dict] = None,
    tifs_min_num=8,
) -> xr.Dataset:
    """
    Will be replaced with dc.load(gm_6month) once they've been produced.
    
    use os.walk to get the all files under a folder, it just merge the half year tifs.
    We need combine two half-year tifs ds and add (calculated indices, rainfall, and slope)
    @param tifs: tifs with the bands
    @param root_fld: the parent folder for the sub_fld
    @param tifs_min_num: geo-median tifs is 16 a tile idx
    @param rename_dict: we can put the rename dictionary here
    @return:
    """
    assert len(tifs) > tifs_min_num
    cache = []
    for tif in tifs:
        if tif.endswith(".tif"):
            band_name = re.search(r"_([A-Za-z0-9]+).tif", tif).groups()[0]
            if band_name in ["rgba", "COUNT"]:
                continue

            band_array = assign_crs(xr.open_rasterio(osp.join(root_fld, tif)).squeeze().to_dataset(name=band_name), crs='epsg:6933')
            cache.append(band_array)
    # clean up output
    output = xr.merge(cache).squeeze()
    output = output.drop(["band"])
    
    return output.rename(rename_dict) if rename_dict else output

def get_tifs_paths(dirname, subfld):
    """
    generated src tifs dictionnary, season on and two, or more seasons
    @param dirname:
    @param subfld:
    @return:
    """
    all_tifs = os.walk(osp.join(dirname, subfld))
    
    return dict(
        (l1_dir, l1_files)
        for level, (l1_dir, _, l1_files) in enumerate(all_tifs)
        if level > 0
    )
