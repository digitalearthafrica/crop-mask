# Feature building code

It is better to start from the configuration in ```src/dea_ai_core/tasks/merge_tifs_to_ds.py```.
We prepared the trained models, rainfall data in the configuration dataclass,

```python
@dataclass
class FeaturePathConfig:
    DATA_PATH = "/g/data/u23/data/"
    REMOTE_PATH = "s3://deafrica-data-dev-af/"
    PRODUCT_NAME = "crop_mask_eastern"
    PRODUCT_VERSION = "v0.1.4"

    TIF_path = osp.join(DATA_PATH, "tifs20")
    model_path = "/g/data/u23/crop-mask/eastern_cropmask/results/gm_mads_two_seasons_ml_model_20210301.joblib"
    model_type = "gm_mads_two_seasons"
    rename_dict = {  # "nir_1": "nir",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B05": "red_edge_1",
        "B06": "red_edge_2",
        "B07": "red_edge_3",
        "B08": "nir",
        "B8A": "nir_narrow",
        "B11": "swir_1",
        "B12": "swir_2",
        "BCMAD": "bcdev",
        "EMAD": "edev",
        "SMAD": "sdev",
    }

    url_slope = "https://deafrica-data.s3.amazonaws.com/ancillary/dem-derivatives/cog_slope_africa.tif"
    rainfall_path = {
        "_S1": "/g/data/CHIRPS/cumulative_alltime/CHPclim_jan_jun_cumulative_rainfall.nc",
        "_S2": "/g/data/CHIRPS/cumulative_alltime/CHPclim_jul_dec_cumulative_rainfall.nc",
    }
    # s1_key, s2_key = "2019-01--P6M", "2019-07--P6M"
    resolution = (-20, 20)
    time = ("2019-01", "2019-12")
    datetime_range = DateTimeRange(time[0], "P12M")
    output_crs = "epsg:6933"
    query = {
        "time": time,
        "resolution": resolution,
        "output_crs": output_crs,
        "group_by": "solar_day",
    }
    training_features = [
        "red_S1",
        "blue_S1",
        "green_S1",
        "nir_S1",
        "swir_1_S1",
        "swir_2_S1",
        "red_edge_1_S1",
        "red_edge_2_S1",
        "red_edge_3_S1",
        "edev_S1",
        "sdev_S1",
        "bcdev_S1",
        "NDVI_S1",
        "LAI_S1",
        "MNDWI_S1",
        "rain_S1",
        "red_S2",
        "blue_S2",
        "green_S2",
        "nir_S2",
        "swir_1_S2",
        "swir_2_S2",
        "red_edge_1_S2",
        "red_edge_2_S2",
        "red_edge_3_S2",
        "edev_S2",
        "sdev_S2",
        "bcdev_S2",
        "NDVI_S2",
        "LAI_S2",
        "MNDWI_S2",
        "rain_S2",
        "slope",
    ]

    prd_properties = {
        "odc:file_format": "GeoTIFF",
        "odc:producer": "digitalearthafrica.org",
        "odc:product": f"{PRODUCT_NAME}",
        "proj:epsg": 6933,
        "crop-mask-model": osp.basename(model_path),
    }
    product = OutputProduct(
        name=PRODUCT_NAME,
        version=PRODUCT_VERSION,
        short_name=PRODUCT_NAME,
        location=REMOTE_PATH,  # place holder
        properties=prd_properties,
        measurements=("mask", "prob"),
        href=f"https://explorer.digitalearth.africa/products/{PRODUCT_NAME}",
    )
```

In the above dataclass definition, we defined

1. root source data path in ```DATA_PATH``` where we save all our raw data and final predictoins

2. remote path ```REMOTE_PATH``` which is normally the s3 bucket with url for the results to be uploaded.

3. product name and product version in ```PRODUCT_NAME``` and ```PRODUCT_VERSION```, where the product name is same
   as we defined in the product yaml file for postgresql database.

4. The extracted tif files will be stored in ```TIF_path``` with the task string as the sub-folder name,
   ```<x>/<y>/<time range>```.

5. The random forest model was trained and saved in ```model_path```. The model type is ```model_type```.

6. The chirps rainfall and slope source path was in ```rainfall_path``` and ```url_slop```.

7. The product was defined through the ```odc.stats.model.OutputProduct``` which should be merged
   with the product yaml later.

#### prepare the source tifs

We prepare the ```tasks``` cache db for source tifs with the command:

```bash
# resolution (-10, 10) first half year of 2019
odc-stats save-tasks --grid africa-10 --temporal_range 2019-01--P6M s2_l2a test1.db
# second half of 2019
odc-stats save-tasks --grid africa-10 --temporal_range 2019-07--P6M s2_l2a test2.db
# also can be done with resolution (-20, 20)
odc-stats save-tasks --grid africa-20 --temporal_range 2019-01--P6M s2_l2a africa-20-2019-01--P6M.db

odc-stats save-tasks --grid africa-20 --temporal_range 2019-01--P6M s2_l2a africa-20-2019-01--P6M.db
```

#### download the geomedian band tifs

Once we collect all tasks cache db, we can start to download the raw geomedian tifs, a working sample
script was prepared in ```src/dea_ai_core/tasks/tasks-exec.py```. Currently, we only processed the eastern region of Africa,
where the tile indices was in the json ```s2_tiles_eastern_aez_tasks.json``` and ```s2_tiles_eastern_aez_tasks_s2.json```.
The command line can be refered to,

```bash
odc-stats run --threads -1 --overwrite --plugin gm-s2 <tifs saving folder> <task cached db> <task str>
```

The task string looks like, ```"x+029/y+000/2019-01--P6M"```.

#### run the prediction pipelines after the tifs were done

In sandbox terminal, we can run the ```src/simple_runner.py``` in the ```src``` folder of the project, which
can run all tiles collected,

```bash
python src/simple_runner.py
```
