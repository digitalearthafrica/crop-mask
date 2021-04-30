import glob
import warnings

warnings.filterwarnings("ignore")
import datacube
import xarray as xr
import geopandas as gpd
import pandas as pd
from datacube.utils.geometry import assign_crs, Geometry
from datacube.utils.rio import configure_s3_access

configure_s3_access(aws_unsigned=True, cloud_defaults=True)

from dea_ml.config.product_feature_config import FeaturePathConfig
from dea_ml.core.feature_layer_default import gm_rainfall_single_season
from datacube.testutils.io import rio_slurp_xarray


def generate_train_data():
    """
    This is a sample pipeline to build base feature,
      - training is hand select polygon
      - prediction is whole time
    :return:
    """
    path = "/home/jovyan/wa/crop-mask/testing/eastern_cropmask/data/Eastern_training_data_20210301.geojson"
    CHIRPS_PREFIX = "/home/jovyan/wa/raw_data/CHIRPS/*.nc"
    # set up our inputs to collect_training_data
    dc = datacube.Datacube(app="feature_build")

    products = ["gm_s2_semiannual"]
    # time = ('2019-01', '2019-12')
    measurements = [
        "red",
        "blue",
        "green",
        "nir",
        "swir_1",
        "swir_2",
        "red_edge_1",
        "red_edge_2",
        "red_edge_3",
        "sdev",
        "bcdev",
        "edev",
    ]
    resolution = (-10, 10)
    output_crs = "epsg:6933"

    # generate a new datacube query object
    query = {
        #     'time': time,
        "product": products[0],
        "measurements": measurements,
        "resolution": resolution,
        "output_crs": output_crs,
        # 'geopolygon': geom,
        "group_by": "solar_day",
        "dask_chunks": {},
    }

    season_time_dict = {"_S1": ("2019-01", "2019-06"), "_S2": ("2019-07", "2019-12")}

    rainfall_dict = dict(
        (
            f"_S{i}",
            assign_crs(xr.open_rasterio(file_name), crs="epsg:4326").drop(
                ["spatial_ref"]
            ),
        )
        for i, file_name in enumerate(sorted(glob.glob(CHIRPS_PREFIX)), 1)
    )

    result_dfs = []

    input_data = gpd.read_file(path)

    for row in input_data.itertuples():
        geom = Geometry(row.geometry, crs="epsg:4326")

        tmp_query = query.copy()
        tmp_query["geopolygon"] = geom

        seasoned_gm = {}
        for season_key in season_time_dict.keys():
            s_gm = gm_rainfall_single_season(
                dc, tmp_query, season_time_dict, rainfall_dict, season_key=season_key
            )

            s_gm = s_gm.rename(
                {var_name: var_name + season_key for var_name in s_gm.data_vars}
            )

            seasoned_gm[season_key] = s_gm

        if seasoned_gm["_S1"].geobox:
            slope = (
                rio_slurp_xarray(
                    FeaturePathConfig.url_slope, gbox=seasoned_gm["_S1"].geobox
                )
                .drop("spatial_ref")
                .to_dataset(name="slope")
            )

            ds = xr.merge(
                [seasoned_gm["_S1"], seasoned_gm["_S2"], slope], compat="override"
            ).chunk({})
            df = ds.to_dataframe().reset_index()
            df["Class"] = row.Class

            result_dfs.append(df)
    output_df = pd.concat(result_dfs)
    output_filename = "/g/data/u23/ml/features/testing.csv"
    output_df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    generate_train_data()
