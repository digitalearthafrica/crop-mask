import logging
from typing import Tuple, List, Dict, Optional

import xarray as xr
from datacube import Datacube
from datacube.testutils.io import rio_slurp_xarray
from dea_ml.core.feature_layer_default import (
    gm_rainfall_single_season,
    merge_two_season_feature,
)
from dea_ml.core.post_processing import post_processing
from dea_ml.core.predict_from_feature import predict_with_model
from dea_ml.helpers.io import read_joblib
from odc.stats import _plugins
from odc.stats.model import DateTimeRange
from odc.stats.model import Task, StatsPluginInterface

_log = logging.getLogger(__name__)


class PredGMS2(StatsPluginInterface):
    """
    Prediction from GeoMAD
    task run with the template
    datakube-apps/src/develop/workspaces/deafrica-dev/processing/06_stats_2019_semiannual_gm.yaml
    """

    source_product = "gm_s2_semiannual"
    target_product = "crop_mask_eastern"

    def __init__(
        self,
        chirps_paths: List[str],
        model_path: str,
        url_slope: str,
        datetime_range: str,
        rename_dict: Dict[str, str],
        training_features: List[str],
        bands: Optional[Tuple] = None,
    ):
        # target band to be saved
        self.chirps_paths = chirps_paths
        self.model_path = model_path
        self.url_slope = url_slope
        self.datetime_range = DateTimeRange(datetime_range)
        self.rename_dict = rename_dict
        self.training_features = training_features
        self.bands = bands if bands else ("mask", "prob", "filtered")

    @property
    def measurements(self) -> Tuple[str, ...]:
        return self.bands

    def input_data(self, task: Task) -> xr.Dataset:
        """
        assemble the input data and do prediction here.
        This method work as pipeline
        """
        dc = Datacube(app=self.target_product)
        ds = dc.load_data(task.datasets, dask_chunks={})
        # ds = dc.load(
        #     product=self.source_product,
        #     time=str(self.datetime_range.start.year),
        #     measurements=list(self.rename_dict.values()),
        #     like=task.geobox,
        #     dask_chunks={},
        # )

        dss = {"_S1": ds.isel(time=0), "_S2": ds.isel(time=1)}

        rainfall_dict = {
            "_S1": rio_slurp_xarray(self.chirps_paths[0]),
            "_S2": rio_slurp_xarray(self.chirps_paths[1]),
        }

        assembled_gm_dict = dict(
            (k, gm_rainfall_single_season(dss[k], rainfall_dict[k])) for k in dss.keys()
        )

        pred_input_data = merge_two_season_feature(assembled_gm_dict, self.url_slope)

        model = read_joblib(self.model_path)
        predicted = predict_with_model(
            self.training_features, model, pred_input_data, {}
        )
        predict, proba, mode = post_processing(predicted, task.geobox)
        output_ds = xr.Dataset({"mask": predict, "prob": proba, "filtered": mode})
        return output_ds

    def reduce(self, xx: xr.Dataset) -> xr.Dataset:
        return xx


_plugins.register("pred-gm-s2", PredGMS2)
